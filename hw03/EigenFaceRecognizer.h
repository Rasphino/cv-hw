//
// Created by rasp on 2019/12/8.
//

#ifndef HW03_EIGENFACERECOGNIZER_H
#define HW03_EIGENFACERECOGNIZER_H

#include <opencv2/opencv.hpp>

#include <vector>

class EigenFaceRecognizer {
public:
  cv::Ptr<EigenFaceRecognizer> create(int num_components = 0, double shreshold = DBL_MAX);
  void train(const std::vector<cv::Mat> &src, const std::vector<int> &labels);
  int predict(const cv::Mat &src);

  double getThreshold() const;
  std::vector<cv::Mat> getProjections() const;
  cv::Mat getLabels() const;
  cv::Mat getEigenValues() const;
  cv::Mat getEigenVectors() const;
  cv::Mat getMean() const;

private:
  int threshold;
  std::vector<cv::Mat> projections;
  std::vector<int> labels;
  cv::Mat eigenvectors;
  cv::Mat eigenvalues;
  cv::Mat mean;
};

#endif // HW03_EIGENFACERECOGNIZER_H
