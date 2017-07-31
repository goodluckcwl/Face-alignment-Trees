#pragma once
#include <opencv2/opencv.hpp>
#include "Utils.h"

cv::Mat	features2(const cv::Mat &image, const int sbin, const int padx, const int pady);

void features(cv::Mat image, int sbin, int padx, int pady, cv::Mat &feat_mat);

//计算滤波器响应。
//每个滤波器对输入特征分别滤波。
std::vector<cv::Mat> fconv(cv::Mat feature, std::vector<cv::Mat> filters);

//计算滤波器相应，用一个滤波器滤波
cv::Mat fconv(cv::Mat feature, cv::Mat filter);
