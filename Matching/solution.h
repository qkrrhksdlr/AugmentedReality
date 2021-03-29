#pragma once
#include <opencv2/opencv.hpp>

void reference(std::vector<cv::Mat>& DataBase);
void matching(cv::String path, std::vector<cv::String> camera, std::vector<cv::Mat>& DataBase);