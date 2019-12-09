#include <iostream>

#include "EigenFaceRecognizer.h"
#include "utils.h"

#include <opencv2/opencv.hpp>

using namespace std;

int main(int argc, char *argv[]) {
  if (argc < 4) {
    cerr << "Wrong arguments!" << endl;
    cerr << "Usage: <program name> <test image> <model file name> <path to JAFFE data>" << endl;
    exit(-1);
  }

  cv::Mat testSample = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
  string model_name = argv[2];
  vector<cv::Mat> images;
  vector<int> labels;
  try {
    utils::read_JAFFE_data(argv[3], images, labels);
  } catch (const cv::Exception &e) {
    cerr << "Error reading JAFFE data: " << e.msg << endl;
    exit(1);
  }


  EigenFaceRecognizer *model = new EigenFaceRecognizer;
  model->read(model_name);

  int id;
  cv::Mat eigenvector;
  int predictedLabel = model->predict(testSample, id, eigenvector);
  string result_message = cv::format("Predicted class = %d.", predictedLabel);
  cout << result_message << endl;


  cv::imshow("origin test image", testSample);
  testSample.convertTo(testSample, CV_64FC1);
//  cv::imshow("test image with most significant eigenvector",
//             utils::norm_0_255(mean.clone().reshape(1, testSample.rows)));
  cv::imshow("similar image", images[id]);
  cv::waitKey(0);
  return 0;
}