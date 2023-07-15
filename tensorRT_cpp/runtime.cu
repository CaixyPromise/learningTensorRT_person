#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "logger.h"
#include "common.h"
#include "buffers.h"
#include "utils/preprocess.h"
#include "utils/postprocess.h"
#include "utils/types.h"

// 读取模型文件的函数
void load_engine_file(const char* engine_file, std::vector<uchar>& engine_data)
{
    // 初始化engine_data，'\0'表示空字符
    engine_data = { '\0' };
    // 打开模型文件，以二进制模式打开
    std::ifstream engine_fp(engine_file, std::ios::binary);
    if (!engine_fp.is_open())	// 如果文件未成功打开，输出错误信息并退出程序
    {
        std::cerr << "Unable to load engine file." << std::endl;
        exit(-1);
    }
    engine_fp.seekg(0, engine_fp.end); // 将文件指针移动到文件末尾，用于获取文件大小
    int length = engine_fp.tellg();	   // 获取文件大小
    engine_data.resize(length);		   // 根据文件大小调整engine_data的大小
    engine_fp.seekg(0, engine_fp.beg); // 将文件指针重新定位到文件开始位置
    // 读取文件内容到engine_data中，reinterpret_cast<char*>是用来将uchar*类型指针转换为char*类型指针
    engine_fp.read(reinterpret_cast<char*> (engine_data.data()), length);
    engine_fp.close();// 关闭文件
}


int main(int argc, char** argv)
{
    if (argc < 3)
    {
        std::cerr << "需要2个参数, 请输入足够的参数, 用法: <engine_file> <input_path_file>" << std::endl;
        return -1;
    }
    // 在推理阶段，我们需要从硬盘上加载优化后的模型，然后执行推理。这个阶段就需要用到IRuntime。
    // 我们首先使用IRuntime的deserializeCudaEngine方法从序列化的数据中加载模型，然后使用加载的模型进行推理。
    const char* engine_file = argv[1];
    const char* input_path_file = argv[2];

    // 1. 创建推理运行时的runtime
    // IRuntime 是 TensorRT 提供的一个接口，主要用于在推理阶段执行序列化的模型。
    // 创建 IRuntime 实例是在推理阶段加载和运行 TensorRT 引擎的首要步骤。
    auto runtime = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(sample::gLogger.getTRTLogger()));
    if (!runtime)
    {
        std::cerr << "Failed to create runtime." << std::endl;
        return -1;
    }

    // 2. 反序列生成engine
    // 加载了保存在硬盘上的模型文件
    // 存储到std::vector<uchar>类型的engine_data变量中，以便于后续的模型反序列化操作。
    std::vector<uchar> engine_data = { '\0' };
    load_engine_file(engine_file, engine_data);
    // 使用IRuntime的deserializeCudaEngine方法将其反序列化为ICudaEngine对象。
    // 在TensorRT中，推理的执行是通过ICudaEngine对象来进行的。
    auto mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(engine_data.data(), engine_data.size()));
    if (!mEngine)
    {
        std::cerr << "Failed to create engine." << std::endl;
        return -1;
    }

    // 3. 创建执行上下文
    // 在 TensorRT 中，推理的执行是通过执行上下文 (ExecutionContext) 来进行的。
    // ExecutionContext 封装了推理运行时所需要的所有信息，包括存储设备上的缓冲区地址、内核参数以及其他设备信息。
    // 因此，当需要在设备上执行模型推理时，ExecutionContext。
    auto context = std::unique_ptr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
    if (!context)
    {
        std::cerr << "Failed to create ExcutionContext." << std::endl;
        return -1;
    }
    // 4. 创建输入输出缓冲区
    // 在 TensorRT 中，BufferManager 是一个辅助类，用于简化输入/输出缓冲区的创建和管理。
    // TensorRT 提供的 BufferManager 类用于简化这个过程，可以自动创建并管理这些缓冲区，使得在进行推理时不需要手动创建和管理这些缓冲区。
    // 当需要在 GPU 上进行推理时，只需要将输入数据放入 BufferManager 管理的缓冲区中，
    // 然后调用推理函数，等待推理完成后，从 BufferManager 管理的缓冲区中获取推理结果即可。
    samplesCommon::BufferManager buffers(mEngine);


    // 5.读入视频
    auto cap = cv::VideoCapture(input_path_file);
    int width = int(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = int(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    int fps = int(cap.get(cv::CAP_PROP_FPS));

    // 写入MP4文件，参数分别是：文件名，编码格式，帧率，帧大小
    cv::VideoWriter writer("./output/test.mp4", cv::VideoWriter::fourcc('H', '2', '6', '4'), fps, cv::Size(width, height));

    cv::Mat frame;
    int frame_index = 0;
    // 申请gpu内存
    cuda_preprocess_init(height * width);


    while (cap.isOpened())
    {
        // 统计运行时间
        auto start = std::chrono::high_resolution_clock::now();

        cap >> frame;
        if (frame.empty())
        {
            break;    
        }
        frame_index++;
        // 输入预处理（实现了对输入图像处理的gpu 加速)
        process_input(frame, (float*)buffers.getDeviceBuffer(kInputTensorName));
        // 6. 执行推理
        context->executeV2(buffers.getDeviceBindings().data());
        // 推理完成后拷贝回缓冲区
        buffers.copyOutputToHost();

        // 从buffer中提取推理结果
        int32_t* num_det = (int32_t*)buffers.getHostBuffer(kOutNumDet); // 目标检测到的个数
        int32_t* cls = (int32_t*)buffers.getHostBuffer(kOutDetCls);     // 目标检测到的目标类别
        float* conf = (float*)buffers.getHostBuffer(kOutDetScores);     // 目标检测的目标置信度
        float* bbox = (float*)buffers.getHostBuffer(kOutDetBBoxes);     // 目标检测到的目标框
        // 非极大值抑制，得到最后的检测框
        std::vector<Detection> bboxs;
        yolo_nms(bboxs, num_det, cls, conf, bbox, kConfThresh, kNmsThresh);

        // 结束时间
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        auto time_str = std::to_string(elapsed) + "ms";
        auto fps_str = std::to_string(1000 / elapsed) + "fps";

        // 绘制结果
        for (size_t i = 0; i < bboxs.size(); i++)
        {
            cv::Rect rect = get_rect(frame, bboxs[i].bbox);
            cv::rectangle(frame, rect, cv::Scalar(0x27, 0xC1, 0x36), 2);
            cv::putText(frame, std::to_string((int)bboxs[i].class_id), cv::Point(rect.x, rect.y - 10), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0x27, 0xC1, 0x36), 2);
        }
        cv::putText(frame, time_str, cv::Point(50, 50), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
        cv::putText(frame, fps_str, cv::Point(50, 100), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
        writer.write(frame);
        
        std::cout << "处理完第" << frame_index << "帧" << std::endl;
        if (cv::waitKey(1) == 27)
            break;
    }
    std::cout << "处理完成!!" << std::endl;
    return 0;
}