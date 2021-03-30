#include "solution.h"
#include <opencv2/opencv.hpp>

void linear_blending1()
{
	cv::Mat img1 = cv::imread("apple.jpg");
	cv::Mat img2 = cv::imread("orange.jpg");

	int width = 512, height = 512;
	cv::resize(img1, img1, cv::Size(width, height));
	cv::resize(img2, img2, cv::Size(width, height));

	cv::Mat mask1 = cv::Mat::zeros(height, width, CV_32F);
	cv::Mat mask2 = cv::Mat::zeros(height, width, CV_32F);
	cv::rectangle(mask1, cv::Rect(0, 0, width, height), cv::Scalar(0.5f), cv::FILLED);
	cv::rectangle(mask2, cv::Rect(0, 0, width, height), cv::Scalar(0.5f), cv::FILLED);

	cv::Mat result = cv::Mat(height, width, CV_8UC3);
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			result.at<cv::Vec3b>(y, x) = img1.at<cv::Vec3b>(y, x) * mask1.at<float>(y, x)
				+ img2.at<cv::Vec3b>(y, x) * mask2.at<float>(y, x);
		}
	}

	cv::imshow("apple", img1);	
	cv::imshow("orange", img2);
	cv::imshow("mask1", mask1);	
	cv::imshow("mask2", mask2);
	cv::imshow("result", result);
	cv::waitKey(0);
}