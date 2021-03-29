#include "solution.h"
#include <opencv2/opencv.hpp>

void matching(cv::String path, std::vector<cv::String> camera, std::vector<cv::Mat>& DataBase)
{
	// Image load
	cv::glob(path, camera, false);

	if (camera.size() == 0)
		std::cout << "No Image" << std::endl;

	std::cout << camera.size() << " size of Query Image load" << std::endl << std::endl;

	// Feature matching
	int percent = 0;
	cv::Ptr<cv::Feature2D> featureExtractor = cv::ORB::create();
	cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");

	for (int query = 0; query < camera.size(); query++)
	{
		std::vector<cv::KeyPoint> Keypoints;
		cv::Mat Descriptors;
		cv::Mat img = cv::imread(camera[query]);
		cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
		cv::resize(img, img, cv::Size(img.cols/4, img.rows/4));

		featureExtractor->detectAndCompute(img, cv::Mat(), Keypoints, Descriptors);

		//std::cout << "<<< " << query + 1 << "th Query Image Matching Start >>>" << std::endl << std::endl;

		int bestNumber = 0, capacity = 0;
		const float ration_thresh = 0.65f;
		std::vector<std::vector<cv::DMatch>> knn_matches;
		std::vector<cv::DMatch> good_matches;

		for (int refer = 0; refer < DataBase.size(); refer++)
		{
			knn_matches.resize(0);
			good_matches.resize(0);

			matcher->knnMatch(Descriptors, DataBase[refer], knn_matches, 2);

			//std::cout << "  * " << refer + 1 << "th DataBase Image Matching " << std::endl;

			for (int i = 0; i < knn_matches.size(); i++)
			{
				if (knn_matches[i][0].distance < ration_thresh * knn_matches[i][1].distance)
					good_matches.push_back(knn_matches[i][0]);
			}

			if (good_matches.size() > capacity)
			{
				capacity = good_matches.size();
				bestNumber = refer;
			}

			//std::cout << "     $ " << refer + 1 << "th good_matches size : " << good_matches.size() << ", bestNumber : " << bestNumber << std::endl << std::endl;
		}

		//std::cout << "    ==> " << query + 1 << "th Query Image is matched with " << bestNumber + 1 << "th DataBase Image" << std::endl << std::endl;

		if (query == bestNumber)
			percent += 1;
	}

	std::cout << "Query Camera Image's Correct Percentage : " << percent % 100 << "%" << std::endl << std::endl;
}