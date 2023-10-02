#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "logger.h"
#include "common.h"
#include "buffers.h"
#include "utils/preprocess.h"
#include "utils/postprocess.h"
#include "utils/types.h"
#include <fstream>
#include "streamer/streamer.hpp"
#include "task/border_cross.h"
#include "task/gather.h"
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include "configParser.hpp"

// 读取模型文件的函数
void load_engine_file(std::string engine_file, std::vector<uchar>& engine_data)
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


class DetectPerson
{
private:
    // 从文件中恢复多边形定点
    void readPoints(std::string filename, Polygon &g_ploygon, int width, int height)
    {
        std::ifstream file(filename);
        if (!file.is_open())
        {
            std::cerr << "Unable to open poly file: " << filename << std::endl;
            exit(-1);
        }
        std::string str;
        while (std::getline(file, str))
        {
            std::stringstream ss(str);
            std::string x, y;
            std::getline(ss, x, ',');
            std::getline(ss, y, ',');

            // recover to original size
            x = std::to_string(std::stof(x) * width);
            y = std::to_string(std::stof(y) * height);

            g_ploygon.push_back({std::stoi(x), std::stoi(y)});
        }
    }

    // 检查是否越界
    void blender_overlay(int x, int y, int radius, cv::Mat &image, float alpha, int height, int width)
    {
        // initial
        int rect_l = x - radius;
        int rect_t = y - radius;
        int rect_w = radius * 2;
        int rect_h = radius * 2;

        int point_x = radius;
        int point_y = radius;

        // check if out of range
        if (x + radius > width)
        {
            rect_w = radius + (width - x);
        }
        if (y + radius > height)
        {
            rect_h = radius + (height - y);
        }
        if (x - radius < 0)
        {
            rect_l = 0;
            rect_w = radius + x;
            point_x = x;
        }
        if (y - radius < 0)
        {
            rect_t = 0;
            rect_h = radius + y;
            point_y = y;
        }
        // get roi
        cv::Mat roi = image(cv::Rect(rect_l, rect_t, rect_w, rect_h));
        cv::Mat color;
        roi.copyTo(color);
        // draw circle
        cv::circle(color, cv::Point(point_x, point_y), radius, cv::Scalar(255, 0, 255), -1);
        // blend
        cv::addWeighted(color, alpha, roi, 1.0 - alpha, 0.0, roi);
    }
private:
    Polygon _g_ploygon;
    std::vector<cv::Point> polygon; // 多边形框
    int _width;
    int _height;
    float _dist_threshold;

public:
    DetectPerson(std::string poly_filenamem,
                  int width, int height, float dist_threshold = 100)
        : _width(width),
          _height(height),
          _dist_threshold(dist_threshold)
    {
        // 加载检查点区域的像素点位置-->readPoints
        readPoints(poly_filenamem, _g_ploygon, _width, _height);
        // 绘制多边形
        for (size_t i = 0; i < _g_ploygon.size(); i++)
        {
            polygon.push_back(cv::Point(_g_ploygon[i].x, _g_ploygon[i].y));
        }
    }

public:
    void detect(cv::Mat& frame, std::vector<Detection>& bboxs)
    {
        // 记录所有的检测框中心点
        std::vector<Point> all_points;
        // 遍历检测结果
        for (size_t j = 0; j < bboxs.size(); j++)
        {
            cv::Rect rect = get_rect(frame, bboxs[j].bbox);
            // 获取检测框中心点
            Point p_center = {rect.x + int(rect.width / 2), rect.y + int(rect.height / 2)};
            // 筛选labelid为0的检测框
            if (bboxs[j].class_id == 0)
            {
                all_points.push_back(p_center);
            }
            // 检测框中心点是否在多边形内，在则画红框，不在则画绿框
            if (isInside(_g_ploygon, p_center))
            {
                cv::rectangle(frame, rect, cv::Scalar(0x00, 0x00, 0xFF), 2);
            }
            else
            {
                cv::rectangle(frame, rect, cv::Scalar(0x27, 0xC1, 0x36), 2);
            }
            // 绘制labelid
            cv::putText(frame, std::to_string((int)bboxs[j].class_id), cv::Point(rect.x, rect.y - 10), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0x27, 0xC1, 0x36), 2);
        }
        // 获取聚集点
        auto gather_points = gather_rule(all_points, _dist_threshold);
        for (size_t i = 0; i < gather_points.size(); i++)
        {

            if (gather_points[i].size() < 3)
                continue;
            for (size_t j = 0; j < gather_points[i].size(); j++)
            {
                // 绘制聚集点
                blender_overlay(gather_points[i][j].x, gather_points[i][j].y, 80, frame, 0.3, _height, _width);
                // cv::circle(frame, cv::Point(gather_points[i][j].x, gather_points[i][j].y), 10, cv::Scalar(0, 0, 255), -1);
            }
        }
        cv::polylines(frame, polygon, true, cv::Scalar(0, 0, 255), 2);
    }
};

class App
{
private:    // 数据结构
    class BufferItem
    {
    public:
        cv::Mat frame;                // 原始图像
        std::vector<Detection> bboxs; // 检测结果
    };
private: // 运行系统环境
    std::shared_ptr<nvinfer1::ICudaEngine> mEngine;
    std::unique_ptr<nvinfer1::IExecutionContext> context;
    cv::VideoCapture cap;
    cv::VideoWriter writer;
    int width, height;
    double fps;
    bool task_status;
    cv::Size frameSize;

private:
    AppConfig _config;
    InputStreamConfig input_stream;
    OutputStreamConfig output_stream;


private:    // 多线程环境锁
    // 每个阶段需要传递的缓存
    std::queue<cv::Mat> input_stream_queue;
    std::queue<BufferItem> inference_queue;
    std::queue<cv::Mat> output_stream_queue;

    // 每个阶段的互斥锁
    std::mutex input_stream_mutex;
    std::mutex inference_mutex;
    std::mutex output_stream_mutex;

    // 每个阶段的not_full条件变量
    std::condition_variable input_stream_NotFull;
    std::condition_variable inference_NotFull;
    std::condition_variable output_stream_NotFull;

    // 每个阶段的not_empty条件变量
    std::condition_variable input_stream_NotEmpty;
    std::condition_variable inference_NotEmpty;
    std::condition_variable output_stream_NotEmpty;
    const int BUFFER_SIZE = 20;

public:
    App(const AppConfig& config_file) 
        : _config(config_file)
        , task_status(false)
    {
        if (__init__())
        {
            std::cout << "系统初始化成功......" << std::endl;
        }
        else {
            std::cout << "初始化失败" << std::endl;
            std::abort();
        }
    }

    // 初始化系统装填
    bool __init__()
    {
        auto runtime = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(sample::gLogger.getTRTLogger()));
        if (!runtime)
        {
            std::cerr << "Failed to create runtime." << std::endl;
            return false;
        }  
        std::vector<uchar> engine_data = { '\0' };
        load_engine_file(_config.engine_file, engine_data);
        mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(engine_data.data(), engine_data.size()));
        if (!mEngine)
        {
            std::cerr << "Failed to create engine." << std::endl;
            return false;
        }

        context = std::unique_ptr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
        if (!context)
        {
            std::cerr << "Failed to create ExcutionContext." << std::endl;
            return false;
        }
        cap = cv::VideoCapture(input_stream.stream_addr);
        std::cout << "输入视频地址: " << input_stream.stream_addr << std::endl;
        if (!cap.isOpened())
        {
            std::cerr << "Failed to open video." << std::endl;
            return -1;
        }
        int width = int(cap.get(cv::CAP_PROP_FRAME_WIDTH));
        int height = int(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
        int fps = int(cap.get(cv::CAP_PROP_FPS));

        // 获取画面尺寸
        frameSize = cv::Size(cap.get(cv::CAP_PROP_FRAME_WIDTH), cap.get(cv::CAP_PROP_FRAME_HEIGHT));

        // 输出视频信息
        std::cout << "视频尺寸: " << frameSize << std::endl;
        std::cout << "视频帧率: " << fps << std::endl;

        return true;
    }
    
    // 读取视频流
    void readFrame()
    {
        cv::Mat frame;
        while (cap.isOpened())
        {
            cap >> frame;
            if (frame.empty())
            {
                task_status = true;
                break;
            }
            // 竞争锁
            std::unique_lock<std::mutex> lock(input_stream_mutex);
            input_stream_NotFull.wait(lock, [this] {return input_stream_queue.size() < BUFFER_SIZE;});
            input_stream_queue.push(frame);
            input_stream_NotEmpty.notify_one();
        }
    }
    
    // 推理
    void inference()
    {
        samplesCommon::BufferManager buffers(mEngine);
        cv::Mat frame;
        int inference_mode = _config.inference_mode;
        std::vector<Detection> bboxs;
        int img_size = frameSize.width * frameSize.height;
        cuda_preprocess_init(img_size);

        while (task_status && !input_stream_queue.empty())
        {
            // 竞争锁, 从输入流队列中取出一个图像
            {
                std::unique_lock<std::mutex> lock(input_stream_mutex);
                input_stream_NotEmpty.wait(lock, [this] {return !input_stream_queue.empty();});
                frame = input_stream_queue.front();
                input_stream_queue.pop();
                input_stream_NotFull.notify_one();
            }
            
            if (inference_mode == 0)    // CPU做完预处理模式
            {
                process_input_cpu(frame, (float *)buffers.getDeviceBuffer(kInputTensorName));
            }
            else if (inference_mode == 1)       // CPU做letterbox，GPU做归一化、BGR2RGB、NHWC to NCHW
            {
                process_input_cv_affine(frame, (float *)buffers.getDeviceBuffer(kInputTensorName));
            }
            else if (inference_mode == 2)   // cuda预处理所有步骤
            {
                process_input_gpu(frame, (float *)buffers.getDeviceBuffer(kInputTensorName));
            }
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
            bboxs.clear();
            yolo_nms(bboxs, num_det, cls, conf, bbox, kConfThresh, kNmsThresh);

            BufferItem item;
            // copy frmae to item
            item.frame = frame.clone();
            // item.frame = frame;
            item.bboxs = std::vector<Detection>(bboxs.begin(), bboxs.end());

            {   // 准备放入后处理处理队列，竞争锁
                std::unique_lock<std::mutex> lock(inference_mutex);
                inference_NotFull.wait(lock, [this] { return inference_queue.size() < BUFFER_SIZE; });
                inference_queue.push(item);
                inference_NotFull.notify_one();
            }
        }
    }
    
    // 图像推理后处理
    void postprocess()
    {
        DetectPerson dtPerson(_config.poly_file, width, height, _config.dist_threshold);
        BufferItem item;
        cv::Mat frame;
        while (task_status && !inference_queue.empty())
        {
            // 竞争锁并取出图像
            {
                std::unique_lock<std::mutex> lock(inference_mutex);
                inference_NotEmpty.wait(lock, [this] { return !inference_queue.empty(); });
                item = inference_queue.front();
                frame = item.frame.clone();
                inference_queue.pop();
                inference_NotFull.notify_one();
            }

            dtPerson.detect(frame, item.bboxs);
            // 推流竞争锁
            {
                std::unique_lock<std::mutex> lock(output_stream_mutex);
                output_stream_NotFull.wait(lock, [this] { return output_stream_queue.size() < BUFFER_SIZE; });
                output_stream_queue.push(frame);
                output_stream_NotEmpty.notify_one();
            }
        }
    }
    
    // 推理结果推流和保存
    void streamer()
    {
        cv::Mat frame;
        streamer::Streamer media_stream;
        bool push_stream = output_stream.push_stream;
        bool write_stream = output_stream.write_stream;
        if (push_stream)
        {
            streamer::StreamerConfig config(frameSize.width, frameSize.height,
                                            frameSize.width, frameSize.height,
                                            fps, output_stream.bitrate,
                                            output_stream.stream_name, 
                                            output_stream.stream_addr);
            int result = media_stream.init(config);
            if (!result)
            {
                std::cout << "初始化推流器成功" << std::endl;
            }
            else 
            {
                std::cout << "初始化推流器失败, 本次以默认视频文件保存" << std::endl;
                push_stream = false;
            }
        }
        
        // 写入MP4文件，参数分别是：文件名，编码格式，帧率，帧大小
        writer = cv::VideoWriter(output_stream.output_file.c_str(), cv::VideoWriter::fourcc('H', '2', '6', '4'), fps, cv::Size(width, height));
        std::cout << "视频输出文件位置: " << output_stream.output_file << std::endl;

        while (task_status && !output_stream_queue.empty())
        {
            // 竞争锁并取出图像
            {
                std::unique_lock<std::mutex> lock(output_stream_mutex);
                output_stream_NotEmpty.wait(lock, [this] { return !output_stream_queue.empty();
                    }
                );
                frame = output_stream_queue.front();
                output_stream_queue.pop();
                output_stream_NotFull.notify_one();
            }
            if (push_stream)
            {
                media_stream.stream_frame(frame.data);
            }
            if (write_stream)
            {
                writer.write(frame);
            }
        }
    }
};


int main(int argc, char** argv)
{
    if (argc < 2)
    {
        std::cerr << "需要2个参数, 请输入足够的参数, 用法: <Config-Yaml File>" << std::endl;
        return -1;
    }
    // 解析配置
    AppConfig app_config(argv[1]);
    app_config.display();

    std::string engine_file = app_config.engine_file;
    InputStreamConfig input_stream = app_config.input_stream;
    OutputStreamConfig output_stream = app_config.output_stream;
    float dist_threshold = app_config.dist_threshold;
    int mode = app_config.inference_mode;
    bool push_stream = output_stream.push_stream;

    // float dist_threshold = std::stof(argv[4]);  // 距离阈值
    // auto push_stream = std::stoi(argv[5]);      // 是否推流
    // auto bitrate = std::stoi(argv[6]);          // 推流比特率
    // // const char* stream_addr = argv[7];          // 推流地址
    // std::string stream_addr(argv[7]);
    // auto mode = std::stoi(argv[8]);             // 模式

    // 在推理阶段，我们需要从硬盘上加载优化后的模型，然后执行推理。这个阶段就需要用到IRuntime。
    // 我们首先使用IRuntime的deserializeCudaEngine方法从序列化的数据中加载模型，然后使用加载的模型进行推理。
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
    auto cap = cv::VideoCapture(input_stream.stream_addr);
    std::cout << "视频地址: " << input_stream.stream_addr << std::endl;
    if (!cap.isOpened())
    {
        std::cerr << "Failed to open video." << std::endl;
        return -1;
    }
    int width = int(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = int(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    int fps = int(cap.get(cv::CAP_PROP_FPS));

    // 写入MP4文件，参数分别是：文件名，编码格式，帧率，帧大小
    cv::VideoWriter writer(output_stream.output_file.c_str(), cv::VideoWriter::fourcc('H', '2', '6', '4'), fps, cv::Size(width, height));
    std::cout << "视频输出地址: " << output_stream.output_file << std::endl;

    cv::Mat frame;
    int frame_index = 0;
    // 获取画面尺寸
    cv::Size frameSize(cap.get(cv::CAP_PROP_FRAME_WIDTH), cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    // 获取帧率
    double video_fps = cap.get(cv::CAP_PROP_FPS);
    // 申请gpu内存
    cuda_preprocess_init(height * width);
    std::vector<Detection> bboxs;
    DetectPerson dtPerson(app_config.poly_file, frameSize.width, frameSize.height, dist_threshold);

    // 初始化推流器
    streamer::Streamer media_stream;

    if (push_stream)
    {
        // 初始化推流器
        std::cout << "初始化推流器" << std::endl;
        streamer::StreamerConfig config(frameSize.width, frameSize.height,
                                                 frameSize.width, frameSize.height,
                                                 video_fps, output_stream.bitrate,
                                                 output_stream.stream_name, 
                                                 output_stream.stream_addr);
        int result = media_stream.init(config);
        if (!result)
        {
            std::cout << "初始化推流器成功" << std::endl;
        }
        else {
            std::cout << "初始化推流器失败, 本次以默认视频文件保存" << std::endl;
            push_stream = false;
        }
    }

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
        // 选择预处理方式
        if (mode == 0) 
        {
            // 使用CPU做letterbox、归一化、BGR2RGB、NHWC to NCHW
            process_input_cpu(frame, (float *)buffers.getDeviceBuffer(kInputTensorName));
        }
        else if (mode == 1)
        {
            // 使用CPU做letterbox，GPU做归一化、BGR2RGB、NHWC to NCHW
            process_input_cv_affine(frame, (float *)buffers.getDeviceBuffer(kInputTensorName));
        }
        else if (mode == 2)
        {
            // 使用cuda预处理所有步骤
            process_input_gpu(frame, (float *)buffers.getDeviceBuffer(kInputTensorName));
        }
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
        bboxs.clear();
        yolo_nms(bboxs, num_det, cls, conf, bbox, kConfThresh, kNmsThresh);

        // 结束时间
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        auto time_str = std::to_string(elapsed) + "ms";
        auto fps_str = std::to_string(1000 / elapsed) + "fps";

        dtPerson.detect(frame, bboxs);
        // 绘制结果
        // for (size_t i = 0; i < bboxs.size(); i++)
        // {
        //     cv::Rect rect = get_rect(frame, bboxs[i].bbox);
        //     cv::rectangle(frame, rect, cv::Scalar(0x27, 0xC1, 0x36), 2);
        //     cv::putText(frame, std::to_string((int)bboxs[i].class_id), cv::Point(rect.x, rect.y - 10), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0x27, 0xC1, 0x36), 2);
        // }
        cv::putText(frame, time_str, cv::Point(50, 50), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
        cv::putText(frame, fps_str, cv::Point(50, 100), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
        if (push_stream)
        {
            media_stream.stream_frame(frame.data);
        }
        writer.write(frame);
        
        std::cout << "处理完第" << frame_index << "帧" << std::endl;
        if (cv::waitKey(1) == 27)
            break;
    }
    std::cout << "处理完成!!" << std::endl;
    return 0;
}