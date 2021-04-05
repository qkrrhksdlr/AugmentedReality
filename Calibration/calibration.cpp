#include <opencv2/opencv.hpp>
#include "main.h"

void calibration()
{
	int imgNum = 13;
	char imgName[256];
	cv::Size patSize = cv::Size(9, 6);
	cv::Size imgSize = cv::Size(640, 480);
	
	// Camera Calibration
	cv::Mat patPt = cv::Mat(patSize.width * patSize.height, 1, CV_32FC3);

	for (int j = 0; j < patSize.height; j++) {
		for (int i = 0; i < patSize.width; i++) {
			cv::Vec3f pt = cv::Vec3f(25.f * i, 25.f * j, 0.0f);
			patPt.at<cv::Vec3f>(patSize.width * j + i) = pt;
		}
	}

	std::vector<cv::Mat> imgPt(imgNum);
	std::vector<cv::Mat> objPt(imgNum, patPt);

	for (int idx = 0; idx < imgNum; idx++) {
		sprintf_s(imgName, "ChessImages/%02d.jpg", idx + 1);
		cv::Mat srcImg = cv::imread(imgName, 1);
		cv::Mat grayImg;
		cv::cvtColor(srcImg, grayImg, CV_BGR2GRAY);

		cv::Mat findPt;
		bool chkPat = cv::findChessboardCorners(grayImg, patSize, imgPt[idx]);

		if (chkPat == true)
			cv::cornerSubPix(grayImg, imgPt[idx], cvSize(11, 11), cvSize(-1, -1), cv::TermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 30, 0.01));
		
		cv::drawChessboardCorners(srcImg, patSize, imgPt[idx], chkPat);

		printf("%d - %d\n", idx + 1, imgNum);
		cv::imshow("Source", srcImg);
		cv::waitKey(0);
	}
	
	cv::Mat camInsMat;
	cv::Mat disMat;
	double chkCal = cv::calibrateCamera(objPt, imgPt, imgSize, camInsMat, disMat, cv::noArray(), cv::noArray());

	printf("// Camera Instrinsic Matrix //\n");
	printf("%.6f, %.6f, %.6f\n", camInsMat.at<double>(0, 0), camInsMat.at<double>(0, 1), camInsMat.at<double>(0, 2));
	printf("%.6f, %.6f, %.6f\n", camInsMat.at<double>(1, 0), camInsMat.at<double>(1, 1), camInsMat.at<double>(1, 2));
	printf("%.6f, %.6f, %.6f\n\n", camInsMat.at<double>(2, 0), camInsMat.at<double>(2, 1), camInsMat.at<double>(2, 2));

	printf("// Distortion Coefficients //\n");
	printf("%.6f, %.6f, %.6f, %.6f, %.6f\n\n", disMat.at<double>(0), disMat.at<double>(1), disMat.at<double>(2), disMat.at<double>(3), disMat.at<double>(4));

	printf("// Reprojection error : %1f //\n", chkCal);
	
	//Undistortion Test
	for (int i = 0; i < imgNum; i++)
	{
		sprintf_s(imgName, "ChessImages/%02d.jpg", i + 1);
		cv::Mat testImg = cv::imread(imgName, 1); 
		cv::Mat undistImg;
		cv::undistort(testImg, undistImg, camInsMat, disMat);

		cv::imshow("SourceImage", testImg);
		cv::imshow("UndistortedImage", undistImg);

		cv::waitKey(0);
		cvDestroyAllWindows();
	}
}
