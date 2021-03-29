#include <opencv2\opencv.hpp>
#include "solution.h"

void warping(cv::Mat& queryImage, cv::Mat& I_image, cv::Mat& H, int width, int height)
{
    /*
    // forward warping
    int height = 833, width = 1026;
    cv::Mat H_image = cv::Mat::zeros(height, width, CV_8UC3);
    for (int y = 0; y < queryImage.rows; y++) {
       for (int x = 0; x < queryImage.cols; x++) {

          cv::Matx<double, 3, 1> P(x, y, 1);
          cv::Mat R = H * P;

          int x2 = R.at<double>(0) / R.at<double>(2) + 717; //721
          int y2 = R.at<double>(1) / R.at<double>(2) + 254;

          H_image.at<cv::Vec3b>(y2, x2) = queryImage.at<cv::Vec3b>(y, x);
       }
    }
    cv::imshow("H_image", H_image);
    */

    // Inverse warping
    cv::Mat Inverse_H = H.inv();
    Inverse_H = Inverse_H / Inverse_H.at<double>(8);

    for (int y = 0; y < I_image.rows; y++) {
        for (int x = 0; x < I_image.cols; x++) {

            cv::Matx<double, 3, 1> P(x - 717, y - 254, 1);

            cv::Mat R = Inverse_H * P;

            float x2 = R.at<double>(0) / R.at<double>(2);
            float y2 = R.at<double>(1) / R.at<double>(2);

            //Bilinear Interpolation
            if (x2 > 0 && y2 > 0 && x2 < 640 && y2 < 480)
            {
                int px1 = (int)x2;
                int py1 = (int)y2;
                int px2 = std::min(px1 + 1, queryImage.cols - 1);
                int py2 = std::min(py1 + 1, queryImage.rows - 1);

                cv::Vec3b P1 = queryImage.at<cv::Vec3b>(py1, px1);
                cv::Vec3b P2 = queryImage.at<cv::Vec3b>(py1, px2);
                cv::Vec3b P3 = queryImage.at<cv::Vec3b>(py2, px1);
                cv::Vec3b P4 = queryImage.at<cv::Vec3b>(py2, px2);

                double fx1 = x2 - px1;
                double fx2 = 1 - fx1;
                double fy1 = y2 - py1;
                double fy2 = 1 - fy1;

                double W1 = fx2 * fy2;
                double W2 = fx1 * fy2;
                double W3 = fx2 * fy1;
                double W4 = fx1 * fy1;

                I_image.at<cv::Vec3b>(y, x) = (W1 * P1) + (W2 * P2) + (W3 * P3) + (W4 * P4);
            }
        }
    }

    // anti-aliasing
    for (int y = 0; y < I_image.rows; y++) {
        for (int x = 1; x < 800; x++) {

            if ((I_image.at<cv::Vec3b>(y, x) == cv::Vec3b(0, 0, 0)) && (I_image.at<cv::Vec3b>(y, x + 1) != cv::Vec3b(0, 0, 0)))
                I_image.at<cv::Vec3b>(y, x) = I_image.at<cv::Vec3b>(y, x + 1) / 2;

            else if ((I_image.at<cv::Vec3b>(y, x) == cv::Vec3b(0, 0, 0)) && (I_image.at<cv::Vec3b>(y, x - 1) != cv::Vec3b(0, 0, 0)))
                I_image.at<cv::Vec3b>(y, x) = I_image.at<cv::Vec3b>(y, x - 1) / 2;
        }
    }

    //cv::imshow("I_image", I_image);
}