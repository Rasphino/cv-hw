#include <iostream>

#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

void fitEllipse(Mat &gray_img, Mat &img, int margin) {
  Mat binary_img;
  Canny(gray_img, binary_img, 230, 230);
  imshow("mar", binary_img);

  vector<vector<Point>> contours;
  findContours(binary_img, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);

  for (const auto &contour : contours) {
    vector<Point> pts;
    for (const auto &p : contour) {
      if (p.x > margin && p.y > margin && p.x < binary_img.cols - margin && p.y < binary_img.rows - margin) {
        pts.push_back(p);
      }
    }

    if (pts.size() < 6)
      continue;
    RotatedRect box = fitEllipse(pts);
    ellipse(img, box, Scalar(0, 0, 255), 1, LINE_AA);
  }

  imshow("result", img);
  waitKey(0);
}

int main(int argc, char *argv[]) {
  if (argc < 2) {
    cerr << "Wrong arguments!" << endl;
    exit(-1);
  }

  Mat gray_img = imread(argv[1], IMREAD_GRAYSCALE);
  Mat img = imread(argv[1]);
  fitEllipse(gray_img, img, 2);
  return 0;
}