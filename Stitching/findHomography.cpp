#include <opencv2\opencv.hpp>
#include "solution.h"

cv::Mat findHomography(cv::Mat& refImage, cv::Mat& queryImage, int distThresh)
{
    cv::Ptr<cv::Feature2D> featureExtractor = cv::ORB::create();
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");

    cv::Mat refDescriptors, queryDescriptors;
    std::vector<cv::KeyPoint> refKeypoints, queryKeypoints;
    std::vector<cv::DMatch> matches;

    featureExtractor->detectAndCompute(refImage, cv::Mat(), refKeypoints, refDescriptors);
    featureExtractor->detectAndCompute(queryImage, cv::Mat(), queryKeypoints, queryDescriptors);
    matcher->match(queryDescriptors, refDescriptors, matches);

    std::vector<cv::DMatch>::iterator it = matches.begin();

    for (; it != matches.end();)
    {
        if (it->distance > distThresh)
            it = matches.erase(it);
        else
            it++;
    }

    /*
    cv::Mat matchingImage;
    cv::drawMatches(queryImage, queryKeypoints, refImage, refKeypoints, matches, matchingImage);
    cv::imshow("matches", matchingImage);
    */

    std::vector<cv::Point2f> query;
    std::vector<cv::Point2f> ref;

    for (std::size_t i = 0; i < matches.size(); i++)
    {
        query.push_back(queryKeypoints[matches[i].queryIdx].pt);
        ref.push_back(refKeypoints[matches[i].trainIdx].pt);
    }

    cv::Mat H(3, 3, CV_32F);
    H = cv::findHomography(query, ref, cv::RANSAC);
    //std::cout << H << std::endl << std::endl;

    return H;
}