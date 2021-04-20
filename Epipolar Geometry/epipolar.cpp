#include <opencv2/opencv.hpp>
#include <random>

constexpr double threshold = 0.000001;

cv::Mat SVD(std::vector<cv::Point2d> random_point1, std::vector<cv::Point2d> random_point2);
void RANSAC(std::vector<cv::DMatch> good_matches, std::vector<cv::KeyPoint> keypoint1, std::vector<cv::KeyPoint> keypoint2,
				std::vector<cv::Point2d>& true_inliers1, std::vector<cv::Point2d>& true_inliers2, cv::Mat& FundamentalMat);
void drawEpiline(cv::Mat img1, cv::Mat img2, std::vector<cv::Point2d> true_inliers1, std::vector<cv::Point2d> true_inliers2, cv::Mat FundamentalMat);

int main()
{
	cv::Mat img1 = cv::imread("img1.png", 0);
	cv::Mat img2 = cv::imread("img2.png", 0);

	// Feature Extract
	cv::Ptr<cv::Feature2D> featureExtractor = cv::ORB::create();
	std::vector<cv::KeyPoint> keypoint1, keypoint2;
	cv::Mat descriptor1, descriptor2;

	featureExtractor->detectAndCompute(img1, cv::noArray(), keypoint1, descriptor1);
	featureExtractor->detectAndCompute(img2, cv::noArray(), keypoint2, descriptor2);

	// KNN Matching
	cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");
	std::vector<std::vector<cv::DMatch>> knn_matches;
	std::vector<cv::DMatch> good_matches;

	matcher->knnMatch(descriptor1, descriptor2, knn_matches, 2);
	for (int i = 0; i < knn_matches.size(); i++)
		if (knn_matches[i][0].distance < knn_matches[i][1].distance * 0.7)
			good_matches.push_back(knn_matches[i][0]);

	// Finding Fundamental matrix using RANSAC
	std::vector<cv::Point2d> true_inliers1, true_inliers2;
	cv::Mat FundamentalMat;

	RANSAC(good_matches, keypoint1, keypoint2, true_inliers1, true_inliers2, FundamentalMat);

	// Drawing Epiline
	drawEpiline(img1, img2, true_inliers1, true_inliers2, FundamentalMat);

	return 0;
}

cv::Mat SVD(std::vector<cv::Point2d> random_point1, std::vector<cv::Point2d> random_point2)
{
	// A matrix
	cv::Mat A(8, 9, CV_64F);
	double* ptr = A.ptr<double>(0);

	for (int j = 0; j < A.rows; j++)
	{
		ptr[j * A.cols + 0] = random_point1[j].x * random_point2[j].x;
		ptr[j * A.cols + 1] = random_point1[j].y * random_point2[j].x;
		ptr[j * A.cols + 2] = random_point2[j].x;
		ptr[j * A.cols + 3] = random_point1[j].x * random_point2[j].y;
		ptr[j * A.cols + 4] = random_point1[j].y * random_point2[j].y;
		ptr[j * A.cols + 5] = random_point2[j].y;
		ptr[j * A.cols + 6] = random_point1[j].x;
		ptr[j * A.cols + 7] = random_point1[j].y;
		ptr[j * A.cols + 8] = 1;
	}

	cv::Mat U, S, Vt;
	cv::SVDecomp(A, S, U, Vt, cv::SVD::Flags::FULL_UV);

	// f matrix
	cv::Mat f(9, 1, CV_64F);
	for (int i = 0; i < 9; i++)
		f.at<double>(i) = Vt.at<double>(8, i);

	// F rank(2) downgrade
	cv::Mat F = f.reshape(0, 3);
	cv::Mat F_U, F_S, F_V;
	cv::SVDecomp(F, F_S, F_U, F_V, cv::SVD::Flags::FULL_UV);
	cv::Mat F_S2 = cv::Mat::zeros(3, 3, CV_64F);

	for (int j = 0; j < F_S2.rows; j++) {
		for (int i = 0; i < F_S2.cols; i++) {
			if (i == j && i != 2 && j != 2)
				F_S2.at<double>(j, i) = F_S.at<double>(i);
		}
	}

	cv::Mat N_F = F_U * F_S2 * F_V;

	return N_F;
}

void RANSAC(std::vector<cv::DMatch> good_matches, std::vector<cv::KeyPoint> keypoint1, std::vector<cv::KeyPoint> keypoint2, 
				std::vector<cv::Point2d> &true_inliers1, std::vector<cv::Point2d> &true_inliers2, cv::Mat &FundamentalMat)
{
	std::vector<cv::Point2d> random_point1, random_point2;
	std::vector<cv::Point2d> inlier_point1, inlier_point2;
	std::vector<cv::Point2d> temp_inliers1, temp_inliers2;

	int max_num = 0, max_loop = 0;
	for (int loop = 0; loop < 10000; loop++)
	{
		int num = 0;
		random_point1.clear();
		random_point2.clear();

		// 8 Point Algorithm
		for (int i = 0; i < 8; i++)
		{
			std::random_device rd;
			std::mt19937_64 rng(rd());
			std::uniform_real_distribution<double> range(0, good_matches.size());

			int random = range(rng);

			random_point1.push_back(keypoint1[good_matches[random].queryIdx].pt);
			random_point2.push_back(keypoint2[good_matches[random].trainIdx].pt);
		}

		// Singular Value Decomposition
		cv::Mat N_F = SVD(random_point1, random_point2); 

		// Computing Matrix & Compare
		for (int i = 0; i < good_matches.size(); i++)
		{
			inlier_point1.push_back(keypoint1[good_matches[i].queryIdx].pt);
			inlier_point2.push_back(keypoint2[good_matches[i].trainIdx].pt);

			cv::Matx<double, 3, 1> p(inlier_point1[i].x, inlier_point1[i].y, 1);
			cv::Matx<double, 3, 1> q(inlier_point2[i].x, inlier_point2[i].y, 1);
			cv::Mat unfiltered = q.t() * N_F * p;

			if (std::abs(unfiltered.at<double>(0)) < threshold)
			{
				num += 1;
				temp_inliers1.push_back(inlier_point1[i]);
				temp_inliers2.push_back(inlier_point2[i]);
			}
		}

		// Update Optimized F
		if (num > max_num)
		{
			max_num = num;
			max_loop = loop;
			FundamentalMat = N_F;

			true_inliers1.clear();
			true_inliers2.clear();

			true_inliers1.assign(temp_inliers1.begin(), temp_inliers1.end());
			true_inliers2.assign(temp_inliers2.begin(), temp_inliers2.end());
		}

		temp_inliers1.clear();
		temp_inliers2.clear();
	}
}

void drawEpiline(cv::Mat img1, cv::Mat img2, std::vector<cv::Point2d> true_inliers1, std::vector<cv::Point2d> true_inliers2, cv::Mat FundamentalMat)
{
	// inlier1 -> P(x, y, 1) , inlier2 -> Q(x, y, 1)
	std::vector<cv::Mat> P;
	std::vector<cv::Mat> Q;

	for (int i = 0; i < true_inliers1.size(); i++)
	{
		cv::Mat p = cv::Mat::ones(3, 1, CV_64F);
		p.at<double>(0) = true_inliers1[i].x;
		p.at<double>(1) = true_inliers1[i].y;
		P.push_back(p);

		cv::Mat q = cv::Mat::ones(3, 1, CV_64F);
		q.at<double>(0) = true_inliers2[i].x;
		q.at<double>(1) = true_inliers2[i].y;
		Q.push_back(q);
	}

	// epilines1 = F.T * Q , epilines2 = F * P
	std::vector<cv::Point3d> epilines1, epilines2;

	for (int i = 0; i < true_inliers1.size(); i++)
	{
		cv::Mat ep1 = FundamentalMat.t() * Q[i];
		cv::Mat ep2 = FundamentalMat * P[i];

		epilines1.push_back(cv::Point3d(ep1));
		epilines2.push_back(cv::Point3d(ep2));
	}

	// Epipolar line
	cv::Mat epImg(std::max(img1.rows, img2.rows), img1.cols + img2.cols, CV_8UC3);
	cv::Rect rect1(0, 0, img1.cols, img1.rows);
	cv::Rect rect2(img1.cols, 0, img2.cols, img2.rows);
	cv::cvtColor(img1, epImg(rect1), CV_GRAY2BGR);
	cv::cvtColor(img2, epImg(rect2), CV_GRAY2BGR);

	cv::RNG rng(0);
	for (int i = 0; i < true_inliers1.size(); i++)
	{
		cv::Scalar color(rng(256), rng(256), rng(256));

		cv::line(epImg(rect2),
			cv::Point(0, -epilines1[i].z / epilines1[i].y),
			cv::Point(img1.cols, -(epilines1[i].z + epilines1[i].x * img1.cols) / epilines1[i].y),
			color);
		cv::circle(epImg(rect1), true_inliers1[i], 3, color, cv::FILLED);

		cv::line(epImg(rect1),
			cv::Point(0, -epilines2[i].z / epilines2[i].y),
			cv::Point(img2.cols, -(epilines2[i].z + epilines2[i].x * img2.cols) / epilines2[i].y),
			color);
		cv::circle(epImg(rect2), true_inliers2[i], 3, color, cv::FILLED);
	}

	cv::imshow("epImg", epImg);
	cv::waitKey(0);
}