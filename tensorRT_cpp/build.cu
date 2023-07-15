#include "NvInfer.h"
#include "NvOnnxParser.h" 
#include "logger.h"
#include "common.h"
#include "buffers.h"
#include "cassert"
#include "utils/config.h"
#include "utils/preprocess.h"
#include "utils/types.h"

// 定义校准数据读取器, 最大最小值校准
// 如果要用熵校准entropy的话改为：IInt8EntropyCalibrator2
class CalibrationDataReader : public nvinfer1::IInt8MinMaxCalibrator
{
private:
    std::string mDataDir;
    std::string mCacheFileName;
    std::vector<std::string> mFileNames;
    int mBatchSize;
    nvinfer1::Dims mInputDims;
    int mInputCount;
    float *mDeviceBatchData { nullptr };
    int mBatchCount;
    int mImgSize;
    int mCurBatch{0};
    std::vector<char> mCalibrationCache;

private:
    void load_dataClassFile(const std::string& filepath)
    {
        std::ifstream ifile(filepath);
        std::string Line;
        while (std::getline(ifile, Line))
        {
            sample::gLogInfo << Line << std::endl;
            mFileNames.push_back(Line);
        }
        mBatchCount = mFileNames.size() / mBatchSize;
        std::cout << "CalibrationDataReader: " << mFileNames.size() 
                  << " images, " << mBatchCount << " batches." << std::endl;
    }

public:
    // 构造函数需要传递的参数包括数据目录、数据列表、BatchSize。
    // 通常会根据模型的需求，初始化输入张量的维度和大小，并在设备上分配相应的内存。
    CalibrationDataReader(const std::string& dataDir, const std::string& filepath, int batchSize = 1)
        : mDataDir(dataDir), mCacheFileName("weights/calibration.cache"),
          mBatchSize(batchSize), mImgSize(kInputH * kInputW)
    {
        mInputDims = {1, 3, kInputH, kInputW};
        mInputCount = mBatchSize * samplesCommon::volume(mInputDims);
        cuda_preprocess_init(mImgSize);
        cudaMalloc(&mDeviceBatchData, kInputH * kInputW * 3 * sizeof(float));
        load_dataClassFile(filepath);
    }

    int32_t getBatchSize() const noexcept override
    {
        return mBatchSize;
    }

    bool getBatch(void* bindings[], const char *names[], int nbBindings) noexcept override
    {
        if (mCurBatch + 1 > mBatchCount)
        {
            return false;
        }
        int offset = kInputW * kInputH * 3 * sizeof(float);
        for (int i = 0; i < mBatchSize; i++)
        {
            int idx = mCurBatch * mBatchSize + i;
            std::string filename = mDataDir + "/" + mFileNames[idx];
            cv::Mat image = cv::imread(filename);
            int new_img_size = image.cols * image.rows;
            if (new_img_size > mImgSize)
            {
                mImgSize = new_img_size;
                cuda_preprocess_destroy();
                cuda_preprocess_init(mImgSize);
            }
            process_input_gpu(image, mDeviceBatchData + i * offset);
        }
        for (int i = 0; i < nbBindings; i++)
        {
            if (!strcmp(names[i], kInputTensorName))
            {
                bindings[i] = mDeviceBatchData + i * offset;
            }
        }
        mCurBatch++;
        return true;
    }

    const void* readCalibrationCache(std::size_t& length) noexcept override
    {
        mCalibrationCache.clear();

        std::ifstream input(mCacheFileName, std::ios::binary);
        input >> std::noskipws;
        if (input.good())
        {
            std::copy(std::istream_iterator<char>(input), std::istream_iterator<char>(),
                      std::back_inserter(mCalibrationCache));
        }
        length = mCalibrationCache.size();
        return length ? mCalibrationCache.data() : nullptr;
    }

    void writeCalibrationCache(const void *cache, std::size_t length) noexcept override
    {
        std::ofstream output(mCacheFileName, std::ios::binary);
        output.write(reinterpret_cast<const char*>(cache), length);
    }
};


int main(int argc, char** argv)
{
    if (argc != 4)
    {
        std::cerr << "请输入onnx文件位置: ./build/[onnx_file] [calib_dir] [calib_list_file]" << std::endl;
        return -1;
    }
    // 命令行获取onnx文件路径、校准数据集路径、校准数据集列表文件
    char* onnx_file = argv[1];
    char* calib_dir = argv[2];
    char* calib_list_file = argv[3];
    // ========== 1. 创建builder：创建优化的执行引擎（ICudaEngine）的关键工具 ==========
    // 在几乎所有使用TensorRT的场合都会使用到IBuilder
    // 只要TensorRT来进行优化和部署，都需要先创建和使用IBuilder。
    std::unique_ptr<nvinfer1::IBuilder> builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger()));
    if (!builder)
    {
        std::cerr << "Failed to create build" << std::endl;
        return -1;
    } 
    std::cout << "Successfully to create builder!!" << std::endl;

    // ========== 2. 创建network：builder--->network ==========
    // 设置batch, 数据输入的批次量大小
    // 显性设置batch
    const unsigned int explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    std::unique_ptr<nvinfer1::INetworkDefinition> network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
    if (!network)
    {
        std::cout << "Failed to create network" << std::endl;
        return -1;
    }

    // 创建onnxparser，用于解析onnx文件
    std::unique_ptr<nvonnxparser::IParser> parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, sample::gLogger.getTRTLogger()));
    // 调用onnxparser的parseFromFile方法解析onnx文件
    bool parsed = parser->parseFromFile(onnx_file, static_cast<int>(sample::gLogger.getReportableSeverity()));
    if (!parsed)
    {
        std::cerr << "Failed to parse onnx file!!" << std::endl;
        return -1;
    }
    // 配置网络参数
    // 我们需要告诉tensorrt我们最终运行时，输入图像的范围，batch size的范围。这样tensorrt才能对应为我们进行模型构建与优化。
    nvinfer1::ITensor* input = network->getInput(0); // 获取了网络的第一个输入节点。
    nvinfer1::IOptimizationProfile* profile = builder->createOptimizationProfile(); // 创建了一个优化配置文件。
    // 网络的输入节点就是模型的输入层，它接收模型的输入数据。
    // 在 TensorRT 中，优化配置文件（Optimization Profile）用于描述模型的输入尺寸和动态尺寸范围。
    // 通过优化配置文件，可以告诉 TensorRT 输入数据的可能尺寸范围，使其可以创建一个适应各种输入尺寸的优化后的模型。

    // 设置最小尺寸
    profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4(1, 3, 640, 640));
    // 设置最优尺寸
    profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4(1, 3, 640, 640));
    // 设置最大尺寸
    profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4(1, 3, 640, 640));

    // ========== 3. 创建config配置：builder--->config ==========
    // 配置解析器
    std::unique_ptr<nvinfer1::IBuilderConfig> config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config)
    {
        std::cout << "Failed to create config" << std::endl;
        return -1;
    }
    // 添加之前创建的优化配置文件（profile）到配置对象（config）中
    // 优化配置文件（profile）包含了输入节点尺寸的设置，这些设置会在模型优化时被使用。
    config->addOptimizationProfile(profile);
    // 设置精度
    if (!builder->platformHasFastInt8())
    {
        sample::gLogInfo << "设备不支持int8，本次将默认使用int16" << std::endl;
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
    }
    else {
        sample::gLogInfo << "设备支持int8，本次将使用int8量化" << std::endl;
        auto calibrator = new CalibrationDataReader(calib_dir, calib_list_file);
        config->setFlag(nvinfer1::BuilderFlag::kINT8);
        config->setInt8Calibrator(calibrator);
    }

    // config->setFlag(nvinfer1::BuilderFlag::kFP16);
    builder->setMaxBatchSize(1);
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1 << 30);

    // 创建流，用于设置profile
    auto profileStream = samplesCommon::makeCudaStream();
    if (!profileStream)
    {
        std::cerr << "Failed to create CUDA profileStream File" << std::endl;
        return -1;
    }
    config->setProfileStream(*profileStream);

    // ========== 5. 序列化保存engine ==========
    // 使用之前创建并配置的 builder、network 和 config 对象来构建并序列化一个优化过的模型。
    std::unique_ptr<nvinfer1::IHostMemory> plan = std::unique_ptr<nvinfer1::IHostMemory>(builder->buildSerializedNetwork(*network, *config));
    std::ofstream engine_file("./weights/best.engine", std::ios::binary);
    assert(engine_file.is_open() && "Failed to open engine file");
    engine_file.write((char *)plan->data(), plan->size());
    engine_file.close();

    // ========== 6. 释放资源 ==========
    sample::gLogInfo << "Engine build success!" << std::endl;
    return 0;
}