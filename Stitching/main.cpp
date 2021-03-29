#include <opencv2\opencv.hpp>
#include "solution.h"

int main()
{
	int height = 833, width = 1026, r_height = 833, r_width = 1400;
	int overlapWidth = 300, x_point = 726, y_point = 255, threshold = 30;

	cv::Mat queryImage = cv::imread("img1.jpg");
	cv::Mat refImage = cv::imread("img2.jpg");

	cv::Mat H = findHomography(refImage, queryImage, threshold);
	cv::Mat I_image = cv::Mat::zeros(height, width, CV_8UC3);
	cv::Mat result = cv::Mat::zeros(r_height, r_width, CV_8UC3);

	warping(queryImage, I_image, H, width, height);
	blending(refImage, queryImage, I_image, result, overlapWidth, x_point, y_point);

	cv::imshow("result", result);
	cv::waitKey(0);
	return 0;
}