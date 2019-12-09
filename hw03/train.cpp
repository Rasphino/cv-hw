#include <iostream>

#include "EigenFaceRecognizer.h"
#include "utils.h"

#include <opencv2/opencv.hpp>

using namespace std;

int main(int argc, char *argv[]) {
  if (argc < 4) {
    cerr << "Wrong arguments!" << endl;
    cerr << "Usage: <program name> <energe> <model file name> <path to JAFFE data>" << endl;
    exit(-1);
  }

  vector<cv::Mat> images;
  vector<int> labels;
  string model_name = argv[2];
  double energy = atof(argv[1]);

  try {
    utils::read_JAFFE_data(argv[3], images, labels);
  } catch (const cv::Exception &e) {
    cerr << "Error reading JAFFE data: " << e.msg << endl;
    exit(1);
  }

  int height = images[0].rows;

  EigenFaceRecognizer *model = new EigenFaceRecognizer;
  model->train(images, labels, energy);

  cv::Mat eigenvalues = model->getEigenValues();
  cv::Mat W = model->getEigenVectors();
  cv::Mat mean = model->getMean();

  imwrite("mean.png", utils::norm_0_255(mean.reshape(1, images[0].rows)));
  cv::Mat eigen = mean.clone();
  for (int i = 0; i < min(10, W.cols); i++) {
    string msg = cv::format("Eigenvalue #%d = %.5f", i, eigenvalues.at<double>(i));
    cout << msg << endl;
    // get eigenvector #i
    cv::Mat ev = W.col(i).clone() + mean;
    eigen += ev;

    cv::Mat grayscale = utils::norm_0_255(ev.reshape(1, height));
    imwrite(cv::format("eigenface_%d.png", i), grayscale);
  }
  cv::Mat grayscale = utils::norm_0_255(eigen.reshape(1, height));
  // Display or save:
  imwrite("eigenface.png", grayscale);

  model->save(model_name);

  return 0;
}