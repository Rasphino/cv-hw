//
// Created by rasp on 2019/12/9.
//

#include "utils.h"

using namespace std;

void utils::read_JAFFE_data(string pathname, vector<cv::Mat> &images, vector<int> &labels) {
  vector<string> files;
  cv::glob(pathname, files);

  int label_id = 0;
  string last_label = files[0].substr(pathname.length() + 1, 2);

  for (const auto &file : files) {
    cv::Mat img = imread(file, cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
      clog << file << " is not a image, skip" << endl;
      continue;
    }
    images.push_back(img);
    string label = file.substr(pathname.length() + 1, 2);
    if (label == last_label) {
      labels.push_back(label_id);
    } else {
      last_label = label;
      labels.push_back(label_id++);
    }
  }
}

cv::Mat utils::norm_0_255(cv::Mat _src) {
  cv::Mat dst;
  switch (_src.channels()) {
  case 1:
    normalize(_src, dst, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    break;
  case 3:
    normalize(_src, dst, 0, 255, cv::NORM_MINMAX, CV_8UC3);
    break;
  default:
    _src.copyTo(dst);
    break;
  }
  return dst;
}

cv::Mat utils::asRowMatrix(const vector<cv::Mat> &src, int rtype) {
  int n = src.size();
  int d = src[0].total();

  cv::Mat data(n, d, rtype);
  for (int i = 0; i < n; ++i) {
    cv::Mat xi = data.row(i);
    if (src[i].isContinuous()) {
      src[i].reshape(1, 1).convertTo(xi, rtype);
    } else {
      src[i].clone().reshape(1, 1).convertTo(xi, rtype);
    }
  }

  return data;
}

cv::Mat utils::asColMatrix(const vector<cv::Mat> &src, int rtype) {
  cv::Mat tmp = asRowMatrix(src, rtype);
  transpose(tmp, tmp);
  return tmp;
}