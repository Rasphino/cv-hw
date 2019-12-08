//
// Created by rasp on 2019/12/8.
//

#ifndef HW03_UTILS_H
#define HW03_UTILS_H

#include <vector>

#include <opencv2/opencv.hpp>

namespace utils {

void read_JAFFE_data(std::string pathname, std::vector<cv::Mat> &images, std::vector<int> &labels);

cv::Mat norm_0_255(cv::Mat _src);

cv::Mat asRowMatrix(const std::vector<cv::Mat> &src, int rtype);

cv::Mat asColMatrix(const std::vector<cv::Mat> &src, int rtype);

} // namespace utils

#endif // HW03_UTILS_H
