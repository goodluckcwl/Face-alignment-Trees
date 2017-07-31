#pragma once
#include <opencv2/opencv.hpp>

void saveMat(const char *fileName,cv::Mat mat);

cv::Mat readMat(const char *fileName,int channels);

cv::Mat compare(cv::Mat a, cv::Mat b,double tolerance);

//遍历矩阵去除padding，保证地址连续
//padx:x方向padding长度
//pady:y方向padding长度
cv::Mat rmPadding(cv::Mat src, int padx, int pady);
