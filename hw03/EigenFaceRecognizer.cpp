//
// Created by rasp on 2019/12/8.
//

#include "EigenFaceRecognizer.h"
#include "utils.h"

using namespace utils;
using namespace std;
using namespace cv;

void EigenFaceRecognizer::train(const std::vector<cv::Mat> &images, const std::vector<int> &labels, double energy) {
  this->labels = labels;

  Mat data = asColMatrix(images, CV_64FC1);

  reduce(data, this->mean, 1, REDUCE_AVG, CV_64FC1);
  Mat delta = data;
  for (int i = 0; i < delta.cols; ++i) {
    cv::Mat xi = data.col(i);
    xi -= this->mean;
  }

  cout << this->mean.size << " " << delta.size << endl;
  Mat cov = 1.0 / images.size() * delta.t() * delta;
  cout << cov.size << endl;

  Mat eigenValues, eigenVector;
  eigen(cov, eigenValues, eigenVector);

  Mat _eigenValueSum;
  double eigenValueSum;
  reduce(eigenValues, _eigenValueSum, 0, REDUCE_SUM, CV_64FC1);
  eigenValueSum = _eigenValueSum.at<double>(0, 0);

  double sum = 0;
  for (int i = 0; i < images.size(); ++i) {
    sum += eigenValues.at<double>(i, 0);

    if (sum / eigenValueSum > energy) {
      this->eigenvalues = eigenValues.rowRange(0, i).clone();
      Mat tmp = delta * eigenVector;
      this->eigenvectors = tmp.colRange(0, i).clone();
      this->projections = this->eigenvectors.t() * delta;
      break;
    }
  }

  cout << this->projections.size << endl;
}

int EigenFaceRecognizer::predict(const cv::Mat &src, int &_id, Mat &proj) {
  Mat data = asColMatrix(src, CV_64FC1) - this->mean;
  cout << (data.type() == CV_64FC1) << " " << (this->mean.type() == CV_64FC1) << endl;

  Mat res = this->eigenvectors.t() * data;
  cout << res << endl;
  cout << res.size << " " << this->projections.col(0).size << " " << (res.type() == CV_64FC1) << " "
       << (this->eigenvectors.type() == CV_64FC1) << endl;

  int id = 0;
  double val = norm(this->projections.col(0), res, NORM_L2);
  cout << projections.size << endl;
  for (int i = 1; i < this->projections.cols; ++i) {
    double tmp = norm(this->projections.col(i), res, NORM_L2);
    if (val > tmp) {
      id = i, val = tmp, proj = this->projections.col(i);
    }
  }
  _id = id;
  return this->labels[id];
}

void EigenFaceRecognizer::save(std::string model_name) {
  FileStorage fs(model_name, FileStorage::WRITE);
  fs << "projections" << projections;
  fs << "labels" << labels;
  fs << "eigenvectors" << eigenvectors;
  fs << "eigenvalues" << eigenvalues;
  fs << "mean" << mean;
}

void EigenFaceRecognizer::read(std::string model_name) {
  FileStorage fs(model_name, FileStorage::READ);
  if (!fs.isOpened()) {
    cerr << "failed to open " << model_name << endl;
    exit(2);
  }

  fs["projections"] >> projections;
  fs["labels"] >> labels;
  fs["eigenvectors"] >> eigenvectors;
  fs["eigenvalues"] >> eigenvalues;
  fs["mean"] >> mean;
}

std::vector<int> EigenFaceRecognizer::getLabels() const { return this->labels; }

cv::Mat EigenFaceRecognizer::getEigenValues() const { return this->eigenvalues; }

cv::Mat EigenFaceRecognizer::getEigenVectors() const { return this->eigenvectors; }

cv::Mat EigenFaceRecognizer::getMean() const { return this->mean; }
