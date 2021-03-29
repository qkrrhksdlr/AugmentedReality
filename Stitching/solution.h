#pragma once

cv::Mat findHomography(cv::Mat& refImage, cv::Mat& queryImage, int distThresh);
void warping(cv::Mat& queryImage, cv::Mat& I_image, cv::Mat& H, int width, int height);
void blending(cv::Mat& refImage, cv::Mat& queryImage, cv::Mat& I_image, cv::Mat& result, int overlapWidth, int x_point, int y_point);