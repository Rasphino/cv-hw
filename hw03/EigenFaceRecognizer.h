//
// Created by rasp on 2019/12/8.
//

#ifndef HW03_EIGENFACERECOGNIZER_H
#define HW03_EIGENFACERECOGNIZER_H

#include <opencv2/opencv.hpp>

#include <vector>

class EigenFaceRecognizer {
public:
  void train(const std::vector<cv::Mat> &src, const std::vector<int> &labels, double energy);
  int predict(const cv::Mat &src, int &id, cv::Mat &proj);

  void save(std::string model_name);
  void read(std::string model_name);

  std::vector<int> getLabels() const;
  cv::Mat getEigenValues() const;
  cv::Mat getEigenVectors() const;
  cv::Mat getMean() const;

private:
  cv::Mat projections;
  std::vector<int> labels;
  cv::Mat eigenvectors;
  cv::Mat eigenvalues;
  cv::Mat mean;
};

#endif // HW03_EIGENFACERECOGNIZER_H
