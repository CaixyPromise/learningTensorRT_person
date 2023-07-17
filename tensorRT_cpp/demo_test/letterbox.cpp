#include <iostream>
#include <opencv2/opencv.hpp>

const int kInputH = 640;
const int kInputW = 640;

// 对输入图片进行letterbox处理，即保持原图比例，将图片放在一个正方形的画布中，多余的部分用黑色填充
// 使用cv::warpAffine，将图片进行仿射变换，将图片放在画布中
// 这里使用inline关键字，表示该函数在编译时会被直接替换到调用处，不会生成函数调用，提高效率
inline cv::Mat letterbox(cv::Mat &src)
{
  // 计算缩放比例
  float scale = std::min(kInputH / (float)src.rows, kInputW / (float)src.cols);
  // 计算偏移量，使得图片放在画布中心
  int offsetx = (kInputW - src.cols * scale) / 2;
  int offsety = (kInputH - src.rows * scale) / 2;

  cv::Point2f srcTri[3]; // 计算原图的三个点：左上角、右上角、左下角
  srcTri[0] = cv::Point2f(0.f, 0.f);
  srcTri[1] = cv::Point2f(src.cols - 1.f, 0.f);
  srcTri[2] = cv::Point2f(0.f, src.rows - 1.f);
  cv::Point2f dstTri[3]; // 计算目标图的三个点：左上角、右上角、左下角
  dstTri[0] = cv::Point2f(offsetx, offsety);
  dstTri[1] = cv::Point2f(src.cols * scale - 1.f + offsetx, offsety);
  dstTri[2] = cv::Point2f(offsetx, src.rows * scale - 1.f + offsety);
  cv::Mat warp_mat = cv::getAffineTransform(srcTri, dstTri); // 计算仿射变换矩阵

  cv::Mat warp_dst = cv::Mat::zeros(kInputH, kInputW, src.type()); // 创建目标图
  cv::warpAffine(src, warp_dst, warp_mat, warp_dst.size());        // 进行仿射变换
  return warp_dst;
}

int main()
{
  cv::Mat src = cv::imread("./test.png");
  // 打印图片shape
  std::cout << "src.rows: " << src.rows << ", src.cols: " << src.cols << ", channels: " << src.channels() << std::endl;

  // resize
  cv::Mat dst;
  cv::resize(src, dst, cv::Size(kInputW, kInputH));
  cv::imwrite("./test_resize.jpg", dst);

  // letterbox
  auto warp_dst = letterbox(src);
  // 再次查看shape
  std::cout << "src.rows: " << warp_dst.rows << ", src.cols: " << warp_dst.cols << ", channels: " << warp_dst.channels() << std::endl;
  // 保存图片
  cv::imwrite("./test_dst.jpg", warp_dst);

  return 0;
}
