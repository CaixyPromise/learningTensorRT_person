extern "C"
{
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
}
#include <iostream>


bool push_stream(const char* input_source, const char* username, const char* password, const char* ip, const char* port, const char* stream_name, bool is_camera) {
    avformat_network_init();

    AVFormatContext* ifmt_ctx = nullptr;
    AVFormatContext* ofmt_ctx = nullptr;
    AVOutputFormat* ofmt = nullptr;
    AVPacket pkt;

    std::string url = "rtsp://";
    url += username;
    url += ":";
    url += password;
    url += "@";
    url += ip;
    url += ":";
    url += port;
    url += "/";
    url += stream_name;

    std::string input_url = is_camera ? "v4l2://" + std::string(input_source) : std::string(input_source);

    int ret = avformat_open_input(&ifmt_ctx, input_url.c_str(), 0, 0);
    if (ret < 0) {
        return false; // Could not open input file
    }

    ret = avformat_find_stream_info(ifmt_ctx, 0);
    if (ret < 0) {
        return false; // Could not retrieve input stream information
    }

    avformat_alloc_output_context2(&ofmt_ctx, nullptr, "rtsp", url.c_str());
    if (!ofmt_ctx) {
        return false; // Could not create output context
    }

    ofmt = ofmt_ctx->oformat;

    for (int i = 0; i < ifmt_ctx->nb_streams; i++) {
        AVStream* in_stream = ifmt_ctx->streams[i];
        AVStream* out_stream = avformat_new_stream(ofmt_ctx, nullptr);
        if (!out_stream) {
            return false; // Failed allocating output stream
        }

        ret = avcodec_parameters_copy(out_stream->codecpar, in_stream->codecpar);
        if (ret < 0) {
            return false; // Failed to copy context from input to output stream codec context
        }
        out_stream->codecpar->codec_tag = 0;
    }

    ret = avio_open(&ofmt_ctx->pb, url.c_str(), AVIO_FLAG_WRITE);
    if (ret < 0) {
        return false; // Could not open output URL
    }

    ret = avformat_write_header(ofmt_ctx, nullptr);
    if (ret < 0) {
        return false; // Error occurred when opening output URL
    }

    while (1) {
        AVStream* in_stream, * out_stream;
        ret = av_read_frame(ifmt_ctx, &pkt);
        if (ret < 0)
            break;

        in_stream = ifmt_ctx->streams[pkt.stream_index];
        out_stream = ofmt_ctx->streams[pkt.stream_index];

        ret = av_interleaved_write_frame(ofmt_ctx, &pkt);
        if (ret < 0) {
            return false; // Error muxing packet
        }

        av_packet_unref(&pkt);
    }

    av_write_trailer(ofmt_ctx);

    avformat_close_input(&ifmt_ctx);
    avio_closep(&ofmt_ctx->pb);
    avformat_free_context(ofmt_ctx);

    return true;
}


// g++ -o output_program main.cpp -I/path/to/ffmpeg/include -L/path/to/ffmpeg/lib -lavformat -lavcodec -lavutil
int main()
{

    return 0;
}