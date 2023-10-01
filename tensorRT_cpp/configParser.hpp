#ifndef CONFIGPARSER_H
#define CONFIGPARSER_H
#include <iostream>
#include <string>
#include <yaml-cpp/yaml.h>

// 输入流配置
class InputStreamConfig
{
public:
    std::string stream_name;
    std::string stream_addr;

public:
    InputStreamConfig()
    {}
    // 全部初始化
    InputStreamConfig(std::string stream_name, 
                      std::string stream_addr)
        : stream_name(stream_name), 
          stream_addr(stream_addr)
    {}
};

// 输出流配置
class OutputStreamConfig
{
public:
    std::string stream_name;
    std::string stream_addr;
    std::string output_file;
    int bitrate;
    bool push_stream;

public:
    OutputStreamConfig()
    {}
    // 全部初始化
    OutputStreamConfig(std::string stream_name,
                       std::string stream_addr,
                       std::string output_file,
                       int bitrate,
                       bool push_stream = true)
        : stream_name(stream_name),
          stream_addr(stream_addr),
          output_file(output_file),
          bitrate(bitrate),
          push_stream(push_stream)
    {}
};

class AppConfig
{
private: // 配置文件
    YAML::Node _config;

public:
    std::string engine_file;
    std::string poly_file;
    InputStreamConfig input_stream;
    OutputStreamConfig output_stream;
    float dist_threshold;    // 距离阈值
    short inference_mode;              // 推理模式
private:
    AppConfig() = default;

private:
    void parser_config()
    {
        if (_config.IsNull())
        {
            std::cerr << "config is null" << std::endl;
            std::abort();
        }
        // 获取engine文件配置
        if (!_config["engine_file"].IsDefined())
        {
            std::cerr << "ValueError: Engine_file is not defined, it's must parameter:)." << std::endl;
            std::abort();
        }
        else
        {
            engine_file = _config["engine_file"].as<std::string>();
        }

        // 获取输入流配置
        if (!_config["input_stream"].IsDefined())
        {
            std::cerr << "ValueError: input_stream is not defined, it's a required parameter:)." << std::endl;
            std::abort();
        }
        else // 解析视频流输入配置
        {
            auto input_stream_node = _config["input_stream"];
            for (const auto& data : input_stream_node) 
            {
                input_stream = InputStreamConfig(data.first.as<std::string>(), data.second.as<std::string>());
            }
        }

        // 获取输出流配置
        if (!_config["output_stream"].IsDefined())
        {
            std::cerr << "ValueError: output_stream is not defined, it's a required parameter:)." << std::endl;
            std::abort();
        }
        else // 解析视频流输出配置
        {
            auto output_stream_node = _config["output_stream"];

            std::string stream_name = "main";
            std::string stream_addr;
            std::string output_file;
            int bitrate = 0;

            if (output_stream_node["file"].IsDefined())
            {
                for (const auto &file : output_stream_node["file"])
                {
                    output_file = file.second.as<std::string>();
                }
            }

            if (output_stream_node["stream"].IsDefined())
            {
                for (const auto &stream : output_stream_node["stream"])
                {
                    stream_addr = stream["main"].as<std::string>();
                    bitrate = stream["bit_rate"].as<int>();
                }
            }
            output_stream = OutputStreamConfig(stream_name, stream_addr, output_file, bitrate);
        }

        dist_threshold = _config["dist_threshold"].as<float>();
        inference_mode = _config["mode"].as<short>();
        poly_file = _config["poly_file"].as<std::string>();
    }

public:
    AppConfig(const YAML::Node &config)
        : _config(config)
    {
        parser_config();
    }

    AppConfig(const std::string& config_file)
    {
        _config = YAML::LoadFile(config_file);
        parser_config();
    }

    AppConfig(const char* config_file)
    {
        _config = YAML::LoadFile(config_file);
        parser_config();
    }

    void display()
    {
        std::cout << "============ App Config ============" << std::endl;
        // 打印其他配置
        std::cout << "Engine File: " << engine_file << std::endl;
        std::cout << "Distance Threshold: " << dist_threshold << std::endl;
        std::cout << "inference_mode: " << inference_mode << std::endl << std::endl;
        std::cout << "polygan File: " << poly_file << std::endl;

        InputStreamConfig input_config = input_stream;
        OutputStreamConfig output_config = output_stream;

        // 打印输入流配置
        std::cout << "Input Stream:" << std::endl;
        if (!input_config.stream_name.empty()) {
            std::cout << "  Stream Name: " << input_config.stream_name << std::endl;
        }
        if (!input_config.stream_addr.empty()) {
            std::cout << "  Stream Address: " << input_config.stream_addr << std::endl;
        }


        // 打印输出流配置
        std::cout << "\nOutput Stream:" << std::endl;
        if (!output_config.stream_name.empty()) {
            std::cout << "  Stream Name: " << output_config.stream_name << std::endl;
        }
        if (!output_config.stream_addr.empty()) {
            std::cout << "  Stream Address: " << output_config.stream_addr << std::endl;
        }
        if (!output_config.output_file.empty()) {
            std::cout << "  File: " << output_config.output_file << std::endl;
        }
        std::cout << "  Bitrate: " << output_config.bitrate << std::endl;
        std::cout << "  Push Stream: " << (output_config.push_stream ? "Yes" : "No") << std::endl;
        std::cout << "============ App Config ============" << std::endl;
    }

};


// int main()
// {
//     std::string config_file = "config.yaml";
//     YAML::Node config = YAML::LoadFile(config_file);
//     // printYAML(config);
//     AppConfig app_config(config);
//     app_config.display();
//     return 0;
// }

#endif // CONFIGPARSER_H