#include "solution.h"
#include <opencv2/opencv.hpp>

void reference(std::vector<cv::Mat>& DataBase)
{
	// Image load
	cv::String ref_path("cd_covers/Reference/*.jpg");
	std::vector<cv::String> Reference;

	cv::glob(ref_path, Reference, false);

	if (Reference.size() == 0)
		std::cout << "No Image" << std::endl;

	// Feature Extract 
	cv::Ptr<cv::Feature2D> featureExtractor = cv::ORB::create();

	for (int index = 0; index < Reference.size(); index++)
	{
		cv::Mat img = cv::imread(Reference[index]);
		cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
		std::vector<cv::KeyPoint> refKeypoints;
		cv::Mat refDescriptors;

		featureExtractor->detectAndCompute(img, cv::Mat(), refKeypoints, refDescriptors);
		DataBase.push_back(refDescriptors);
	}

	std::cout << "Reference Image Feature DataBase Complete !" << std::endl << std::endl;
}