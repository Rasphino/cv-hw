#include <iostream>
#include <string>

#include <opencv2/freetype.hpp>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

void loadImages(const string &dirname, vector<Mat> &img_list, bool showImages = false) {
  /* load all images file in a directory to a vector<Mat> */
  vector<string> files;
  glob(dirname, files);

  for (const auto &file : files) {
    Mat img = imread(file);
    if (img.empty()) {
      clog << file << " is invalid, skip" << endl;
      continue;
    }
    if (showImages) {
      imshow("image", img);
      waitKey(0);
    }
    img_list.push_back(img);
  }
}

bool loadVideo(const string &dirname, VideoCapture &video) {
  /* load first found *.avi video */
  vector<string> files;
  glob(dirname, files);

  for (const auto &file : files) {
    if (file.find(".avi") != string::npos) {
      video.open(file);
      return video.isOpened();
    }
  }
  return false;
}

void resizeKeepAspectRatio(const Mat &input, Mat &output, const Size &dstSize, const Scalar &bgColor) {
  Mat res;
  double h1 = dstSize.width * (input.rows / (double)input.cols);
  double w2 = dstSize.height * (input.cols / (double)input.rows);
  if (h1 <= dstSize.height) {
    resize(input, res, Size(dstSize.width, h1));
  } else {
    resize(input, res, Size(w2, dstSize.height));
  }

  int top = (dstSize.height - res.rows) / 2;
  int down = (dstSize.height - res.rows + 1) / 2;
  int left = (dstSize.width - res.cols) / 2;
  int right = (dstSize.width - res.cols + 1) / 2;
  copyMakeBorder(res, res, top, down, left, right, BORDER_CONSTANT, bgColor);

  output = res;
}

int main(int argc, char *argv[]) {
  if (argc < 2) {
    cerr << "Wrong arguments!" << endl;
    exit(-1);
  }

  // read video and images assets from directory
  vector<Mat> img_list;
  VideoCapture video;
  loadImages(argv[1], img_list);
  if (!loadVideo(argv[1], video) || img_list.empty()) {
    cerr << "Failed to read video/images in directory: " << argv[1] << endl;
    exit(-1);
  }

  // get video info - fps, height, width
  double frame_rate = video.get(CAP_PROP_FPS);
  int height = static_cast<int>(video.get(CAP_PROP_FRAME_HEIGHT));
  int width = static_cast<int>(video.get(CAP_PROP_FRAME_WIDTH));
  clog << frame_rate << " " << height << " " << width << endl;

  Mat frame;
  for (const auto &img : img_list) {
    resizeKeepAspectRatio(img, frame, Size(width, height), Scalar::all(0));
    imshow("Movie Player", frame);
    waitKey(1500);
  }

  video.set(CAP_PROP_POS_FRAMES, 0);
  int i = 0;
  while (i++, 1) {
    if (!video.read(frame))
      break;

    putText(frame, "hello world", Point(i, 2 * i), FONT_HERSHEY_SIMPLEX, 1, Scalar::all(255));
    imshow("Movie Player", frame);

    char c = waitKey(33);
    if (c == 27)
      break;
  }
  video.release();
  destroyWindow("Movie Player");

  return 0;
}
