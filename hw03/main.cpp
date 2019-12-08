#include <iostream>

#include "EigenFaceRecognizer.h"
#include "utils.h"

#include <opencv2/face.hpp>
#include <opencv2/opencv.hpp>

using namespace std;

int main(int argc, char *argv[]) {
  if (argc < 2) {
    cerr << "Wrong arguments!" << endl;
    exit(-1);
  }

  vector<cv::Mat> images;
  vector<int> labels;

  try {
    utils::read_JAFFE_data(argv[1], images, labels);
  } catch (const cv::Exception &e) {
    cerr << "Error reading JAFFE data: " << e.msg << endl;
    exit(1);
  }

  int height = images[0].rows;

  cv::Mat testSample = images[images.size() - 1];
  int testLabel = labels[labels.size() - 1];
  images.pop_back();
  labels.pop_back();

  cv::Ptr<cv::face::EigenFaceRecognizer> model = cv::face::EigenFaceRecognizer::create();
  model->train(images, labels);
//  EigenFaceRecognizer model;
//  model.train(images, labels);

  int predictedLabel = model->predict(testSample);
  string result_message = cv::format("Predicted class = %d / Actual class = %d.", predictedLabel, testLabel);
  cout << result_message << endl;

  cv::Mat eigenvalues = model->getEigenValues();
  cv::Mat W = model->getEigenVectors();
  cv::Mat mean = model->getMean();

  imshow("mean", utils::norm_0_255(mean.reshape(1, images[0].rows)));
  for (int i = 0; i < min(10, W.cols); i++) {
    string msg = cv::format("Eigenvalue #%d = %.5f", i, eigenvalues.at<double>(i));
    cout << msg << endl;
    // get eigenvector #i
    cv::Mat ev = W.col(i).clone();
    // Reshape to original size & normalize to [0...255] for imshow.
    cv::Mat grayscale = utils::norm_0_255(ev.reshape(1, height));
    // Display or save:
    imshow(cv::format("eigenface_%d", i), grayscale);
  }

  for (int num_components = min(W.cols, 10); num_components < min(W.cols, 300); num_components += 15) {
    // slice the eigenvectors from the model
    cv::Mat evs = cv::Mat(W, cv::Range::all(), cv::Range(0, num_components));
    cv::Mat projection = cv::LDA::subspaceProject(evs, mean, images[0].reshape(1, 1));
    cv::Mat reconstruction = cv::LDA::subspaceReconstruct(evs, mean, projection);
    // Normalize the result:
    reconstruction = utils::norm_0_255(reconstruction.reshape(1, images[0].rows));
    // Display or save:-
    imshow(cv::format("eigenface_reconstruction_%d", num_components), reconstruction);
  }

  cv::waitKey(0);
  return 0;
}