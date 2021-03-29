#include <opencv2\opencv.hpp>
#include "solution.h"

void blending(cv::Mat& refImage, cv::Mat& queryImage, cv::Mat& I_image, cv::Mat& result, int overlapWidth, int x_point, int y_point)
{
    //Mask
    cv::Mat mask1 = cv::Mat::zeros(I_image.rows, I_image.cols, CV_32F);
    cv::Mat mask2 = cv::Mat::zeros(refImage.rows, refImage.cols, CV_32F);

    cv::rectangle(mask1, cv::Rect(0, 0, I_image.cols, I_image.rows), cv::Scalar(1.0f), cv::FILLED);
    cv::rectangle(mask1, cv::Rect(x_point, y_point, overlapWidth, refImage.rows), cv::Scalar(0.5f), cv::FILLED);
    cv::rectangle(mask2, cv::Rect(0, 0, overlapWidth, refImage.rows), cv::Scalar(0.5f), cv::FILLED);
    cv::rectangle(mask2, cv::Rect(overlapWidth, 0, (refImage.cols - overlapWidth), refImage.rows), cv::Scalar(1.0f), cv::FILLED);

    for (int ox = overlapWidth, x = x_point; ox > 0; ox--, x++) {
        for (int y = 0; y < refImage.rows; y++) {
            mask1.at<float>(y + y_point, x) = (1.0f / overlapWidth) * (ox);
        }
    }

    for (int ox = 0, x = 0; ox < overlapWidth; ox++, x++) {
        for (int y = 0; y < refImage.rows; y++) {
            mask2.at<float>(y, x) = (1.0f / overlapWidth) * (ox);
        }
    }

    //cv::imshow("mask1", mask1);
    //cv::imshow("mask2", mask2);

    //Alpha Blending
    for (int y = 0; y < I_image.rows; y++) {
        for (int x = 0; x < I_image.cols; x++) {
            result.at<cv::Vec3b>(y, x) = I_image.at<cv::Vec3b>(y, x) * mask1.at<float>(y, x);
        }
    }

    for (int y = 0; y < refImage.rows; y++) {

        for (int x = 0; x < overlapWidth; x++) {

            if (I_image.at<cv::Vec3b>(y + y_point, x + x_point) != cv::Vec3b(0, 0, 0))
                result.at<cv::Vec3b>(y + y_point, x + x_point) += refImage.at<cv::Vec3b>(y, x) * mask2.at<float>(y, x);
            else
                result.at<cv::Vec3b>(y + y_point, x + x_point) += refImage.at<cv::Vec3b>(y, x);
        }

        for (int x = overlapWidth; x < refImage.cols; x++) {
            result.at<cv::Vec3b>(y + y_point, x + x_point) += refImage.at<cv::Vec3b>(y, x);
        }

    }
}