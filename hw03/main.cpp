#include <fstream>
#include <iostream>
#include <sstream>

#include <opencv2/face.hpp>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

Mat norm_0_255(Mat _src) {
  Mat dst;
  switch (_src.channels()) {
  case 1:
    normalize(_src, dst, 0, 255, NORM_MINMAX, CV_8UC1);
    break;
  case 3:
    normalize(_src, dst, 0, 255, NORM_MINMAX, CV_8UC3);
    break;
  default:
    _src.copyTo(dst);
    break;
  }
  return dst;
}

void read_JAFFE_data(string pathname, vector<Mat> &images, vector<int> &labels) {
  vector<string> files;
  glob(pathname, files);

  int label_id = 0;
  string last_label = files[0].substr(pathname.length() + 1, 2);

  for (const auto &file : files) {
    Mat img = imread(file, IMREAD_GRAYSCALE);
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

int main(int argc, char *argv[]) {
  if (argc < 2) {
    cerr << "Wrong arguments!" << endl;
    exit(-1);
  }

  vector<Mat> images;
  vector<int> labels;

  try {
    read_JAFFE_data(argv[1], images, labels);
  } catch (const Exception &e) {
    cerr << "Error reading JAFFE data: " << e.msg << endl;
    exit(1);
  }

  int height = images[0].rows;

  Mat testSample = images[images.size() - 1];
  int testLabel = labels[labels.size() - 1];
  images.pop_back();
  labels.pop_back();

  Ptr<face::EigenFaceRecognizer> model = face::EigenFaceRecognizer::create();
  model->train(images, labels);

  int predictedLabel = model->predict(testSample);
  string result_message = format("Predicted class = %d / Actual class = %d.", predictedLabel, testLabel);
  cout << result_message << endl;

  Mat eigenvalues = model->getEigenValues();
  Mat W = model->getEigenVectors();
  Mat mean = model->getMean();

  imshow("mean", norm_0_255(mean.reshape(1, images[0].rows)));
  for (int i = 0; i < min(10, W.cols); i++) {
    string msg = format("Eigenvalue #%d = %.5f", i, eigenvalues.at<double>(i));
    cout << msg << endl;
    // get eigenvector #i
    Mat ev = W.col(i).clone();
    // Reshape to original size & normalize to [0...255] for imshow.
    Mat grayscale = norm_0_255(ev.reshape(1, height));
    // Display or save:
    imshow(format("eigenface_%d", i), grayscale);
  }

  for (int num_components = min(W.cols, 10); num_components < min(W.cols, 300); num_components += 15) {
    // slice the eigenvectors from the model
    Mat evs = Mat(W, Range::all(), Range(0, num_components));
    Mat projection = LDA::subspaceProject(evs, mean, images[0].reshape(1, 1));
    Mat reconstruction = LDA::subspaceReconstruct(evs, mean, projection);
    // Normalize the result:
    reconstruction = norm_0_255(reconstruction.reshape(1, images[0].rows));
    // Display or save:-
    imshow(format("eigenface_reconstruction_%d", num_components), reconstruction);
  }

  waitKey(0);
  return 0;
}