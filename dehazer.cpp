#include <math.h>

#include <algorithm>
#include <iostream>
#include <numeric>
#include <opencv2/opencv.hpp>
#include <vector>

using namespace cv;
using namespace std;

vector<int> argsort(const vector<float> &array) {
  vector<int> indices(array.size());
  iota(indices.begin(), indices.end(), 0);
  sort(indices.begin(), indices.end(), [&array](int left, int right) -> bool {
    // sort indices according to corresponding array element
    return array[left] < array[right];
  });

  return indices;
}

void darkChannel(Mat &img, Mat &darkdst) {
  Mat bands[3];
  split(img, bands);
  Size a(15, 15);
  Mat dc = min(min(bands[0], bands[1]), bands[2]);
  Mat kernel;
  getStructuringElement(MORPH_RECT, a).convertTo(kernel, CV_32F);
  erode(dc, darkdst, kernel);
}

void AtmLight(Mat &img, float A[], Mat &dark) {
  int imgsz = img.rows * img.cols;
  int numpx = int(max(floor(imgsz / 1000), 1.0));
  Mat darkvec, imvec;
  darkvec.convertTo(darkvec, CV_32F);
  imvec.convertTo(imvec, CV_32F);
  darkvec = dark.reshape(0, imgsz);
  imvec = img.reshape(3, imgsz);
  vector<float> darkvec1(darkvec.rows * darkvec.cols * darkvec.channels());
  if (darkvec.isContinuous()) {
    darkvec1.assign(darkvec.data,
                    darkvec.data + darkvec.total() * darkvec.channels());
  }
  vector<int> indices;
  indices = argsort(darkvec1);
  auto iterator = indices.begin();
  indices.erase(iterator, iterator + imgsz - numpx);

  float atmsum[3] = {};
  for (int i = 1; i < numpx; i++) {
    for (int j = 0; j < 3; j++) {
      atmsum[j] = atmsum[j] + imvec.at<Vec3b>(indices[i])[j];
    }
  }

  for (int z = 0; z < 3; z++) {
    A[z] = atmsum[z] / (numpx * 255);
  }
}

void transmissionEstimate(Mat &img, float A[], Mat &trans) {
  float omega = .95;
  Mat im3;
  im3.convertTo(im3, CV_32F);
  Mat bands[3];
  split(img, bands);
  for (int i = 0; i < 3; i++) {
    bands[i].convertTo(bands[i], CV_32F, 1.0 / (A[i]));
  }
  merge(bands, 3, im3);
  Mat dark;
  dark.convertTo(dark, CV_32F);
  darkChannel(im3, dark);
  dark.convertTo(trans, -1, omega);
  trans = 1 - trans;
}

void guidedFilter(Mat &img, Mat &trans, int r, float eps, Mat &filter) {
  Mat mean_I, mean_p, meanIp;
  meanIp = img.mul(trans);
  Size a(r, r);
  boxFilter(img, mean_I, CV_64F, a);
  boxFilter(trans, mean_p, CV_64F, a);
  boxFilter(meanIp, meanIp, CV_64F, a);
  Mat cov_Ip = mean_I.mul(mean_p);
  cov_Ip = meanIp - cov_Ip;
  Mat mean_II;
  boxFilter(img.mul(img), mean_II, CV_64F, a);
  Mat var_I = mean_I.mul(mean_I);
  var_I = mean_II - var_I;
  Mat a2 = var_I + eps;
  a2 = cov_Ip / a2;
  Mat b = a2.mul(mean_I);
  b = mean_p - b;
  Mat mean_a, mean_b;
  boxFilter(a2, mean_a, CV_64F, a);
  boxFilter(b, mean_b, CV_64F, a);
  Mat imgplace;
  img.convertTo(imgplace, CV_64F);
  mean_a = mean_a.mul(imgplace);
  filter = mean_a + mean_b;
}

void transmissionRefine(Mat &img, Mat &trans, Mat &filter) {
  Mat gray;
  cvtColor(img, gray, COLOR_BGR2GRAY);
  gray.convertTo(gray, CV_32F, 1.0 / 255);
  int r = 60;
  float eps = .0001;
  guidedFilter(gray, trans, r, eps, filter);
}

void recover(Mat &img, Mat &trans, float A[], Mat &im3) {
  trans.convertTo(trans, CV_32F);
  Mat bands[3];
  split(img, bands);
  for (int i = 0; i < 3; i++) {
    bands[i] = bands[i] - A[i];
    bands[i] = bands[i] / trans;
    bands[i] = bands[i] + A[i];
  }
  merge(bands, 3, im3);
}

int main(int argc, char **argv) {
  Mat img = imread("low.png");
  Mat imgCopy;
  bitwise_not(img, img);
  img.convertTo(imgCopy, CV_32F, (1.0 / 255), 0);
  Mat darkdst;
  darkdst.convertTo(darkdst, CV_32F);
  Mat trans;
  Mat filter;
  Mat fin;
  // darkdst.convertTo(darkdst, CV_32FC3, 1.0 / 255);
  float A[3] = {};
  darkChannel(img, darkdst);
  AtmLight(img, A, darkdst);
  transmissionEstimate(imgCopy, A, trans);
  transmissionRefine(img, trans, filter);
  recover(imgCopy, filter, A, fin);
  fin.convertTo(fin, CV_16S, 255);
  threshold(fin, fin, 0, 0, THRESH_TOZERO);
  threshold(fin, fin, 255, 0, THRESH_TRUNC);
  fin.convertTo(fin, CV_8U);
  bitwise_not(fin, fin);

  //   img.convertTo(img, CV_8UC3, 255);
  //  darkdst.convertTo(darkdst, CV_8U, 255);
  namedWindow("Display Image OG", WINDOW_AUTOSIZE);
  imshow("Display Image OG", imgCopy);
  namedWindow("Display Image Dark", WINDOW_AUTOSIZE);
  imshow("Display Image Dark", darkdst);
  namedWindow("Display Image TE", WINDOW_AUTOSIZE);
  imshow("Display Image TE", trans);
  namedWindow("Display Image filter", WINDOW_AUTOSIZE);
  imshow("Display Image filter", filter);
  namedWindow("New Image", WINDOW_NORMAL);
  imshow("New Image", fin);
  waitKey(0);
  return 0;
}
