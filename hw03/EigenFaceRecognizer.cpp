//
// Created by rasp on 2019/12/8.
//

#include "EigenFaceRecognizer.h"
#include "utils.h"

using namespace utils;
using namespace std;
using namespace cv;

void EigenFaceRecognizer::train(const std::vector<cv::Mat> &images, const std::vector<int> &labels) {
  Mat data = asColMatrix(images, CV_64FC1);

  Mat avgData;
  reduce(data, avgData, 1, REDUCE_AVG, CV_64FC1);

  this->mean = norm_0_255(avgData.reshape(0, images[0].rows));
}

int EigenFaceRecognizer::predict(const cv::Mat &src) { return 0; }
