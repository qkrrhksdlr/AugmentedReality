#include <opencv2/opencv.hpp>
#include "main.h"

void camera()
{
	cv::VideoCapture cap(0, cv::CAP_DSHOW);
	cap.set(cv::CAP_PROP_AUTOFOCUS, 0);

	if (!cap.isOpened())
	{
		std::cout << "Can't open the camera" << std::endl;
	}

	int count = 1;
	char buf[256];
	cv::Mat img;

	std::cout << "camera start !" << std::endl;

	while (1)
	{
		cap >> img;
		imshow("camera img", img);
		
		char key = -1;
		char ch = cv::waitKey(10); if (ch != -1) key = ch;
		
		if (key == 27) break;

		if (key == 32)
		{
			sprintf_s(buf, "./ChessImages/%02d.jpg", count); 
			cv::imwrite(buf, img);

			std::cout << count << ".jpg captured !" << std::endl;
			count++;
		}
	}

	std::cout << "camera finished !" << std::endl;

}