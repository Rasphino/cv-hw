#include <iostream>
#include <string>

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

  VideoWriter video_output("output.avi", VideoWriter::fourcc('H', '2', '6', '4'), frame_rate, Size(width, height));

  Mat frame, black(height, width, CV_8UC3, Scalar::all(0));
  Point caption_pos = {width / 2, height / 2};
  Point caption_speed = {3, 4};
  string caption = "3170105166 - Honghao Li";
  int animate_duration = 0.8 * static_cast<int>(frame_rate);

  for (const auto &img : img_list) {
    resizeKeepAspectRatio(img, frame, Size(width, height), Scalar::all(0));
    putText(frame, caption, Point(width / 2 - caption.length() * 10, height - 20), FONT_HERSHEY_SIMPLEX, 1,
            Scalar(0, 0, 255));

    for (int i = 0; i < animate_duration; ++i) {
      Mat tmp;
      double ratio = double(i) / animate_duration;
      addWeighted(frame, ratio, black, 1 - ratio, 0.0, tmp);
      video_output.write(tmp);
    }

    for (int i = 0; i < 1.8 * static_cast<int>(frame_rate); ++i) {
      video_output.write(frame);
    }

    for (int i = 0; i < animate_duration; ++i) {
      Mat tmp;
      double ratio = double(i) / animate_duration;
      addWeighted(frame, 1 - ratio, black, ratio, 0.0, tmp);
      video_output.write(tmp);
    }
  }

  video.set(CAP_PROP_POS_FRAMES, 0);
  while (video.read(frame)) {
    caption_pos = caption_pos + caption_speed;
    if (caption_pos.x < 2 * caption.length() * 10 || caption_pos.x > width - 2 * caption.length() * 10) caption_speed.x *= -1;
    if (caption_pos.y < 20 || caption_pos.y > height - 20) caption_speed.y *= -1;
    putText(frame, caption, Point(caption_pos), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255));
    video_output.write(frame);
  }
  video.release();

  return 0;
}
