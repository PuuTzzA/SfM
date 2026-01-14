#include "detect.hpp"
#include <iostream>

namespace SfM::detect
{
    void harrisCornerDetectionOpenCv(cv::Mat &image)
    {
        cv::Mat gray;
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

        gray.convertTo(gray, CV_32FC1);

        cv::Mat dst;
        cv::cornerHarris(gray, dst, 2, 3, 0.04);

        // Result is dilated for marking the corners
        cv::dilate(dst, dst, cv::Mat());

        double maxVal;
        cv::minMaxLoc(dst, nullptr, &maxVal);

        // Threshold and mark corners in red
        for (int y = 0; y < dst.rows; ++y)
        {
            for (int x = 0; x < dst.cols; ++x)
            {
                if (dst.at<float>(y, x) > 0.01 * maxVal)
                {
                    image.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 255);
                }
            }
        }

        cv::imwrite("../../Data/corner.png", image);

        /* cv::imshow("dst", image);

        if (cv::waitKey(0) == 27)
        {
            cv::destroyAllWindows();
        } */
    };

    void harrisCornerDetectionSubPixelOpenCv(cv::Mat &image)
    {
        cv::Mat gray;
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

        // Find Harris corners
        gray.convertTo(gray, CV_32FC1);

        cv::Mat dst;
        cv::cornerHarris(gray, dst, 2, 3, 0.04);
        cv::dilate(dst, dst, cv::Mat());

        double maxVal;
        cv::minMaxLoc(dst, nullptr, &maxVal);

        cv::Mat dstThresh;
        cv::threshold(dst, dstThresh, 0.01 * maxVal, 255, cv::THRESH_BINARY);
        dstThresh.convertTo(dstThresh, CV_8UC1);

        // Find centroids
        cv::Mat labels, stats, centroids;
        cv::connectedComponentsWithStats(dstThresh, labels, stats, centroids);

        // Define termination criteria
        cv::TermCriteria criteria(
            cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER,
            100,
            0.001);

        // Refine corner locations
        cv::Mat corners;
        centroids.convertTo(corners, CV_32FC2);
        cv::cornerSubPix(gray, corners, cv::Size(5, 5), cv::Size(-1, -1), criteria);

        // Draw results
        for (int i = 0; i < centroids.rows; ++i)
        {
            cv::Point c0(
                static_cast<int>(centroids.at<double>(i, 0)),
                static_cast<int>(centroids.at<double>(i, 1)));

            cv::Point c1(
                static_cast<int>(corners.at<cv::Point2f>(i).x),
                static_cast<int>(corners.at<cv::Point2f>(i).y));

            image.at<cv::Vec3b>(c0) = cv::Vec3b(0, 0, 255); // red
            image.at<cv::Vec3b>(c1) = cv::Vec3b(0, 255, 0); // green
        }

        cv::imwrite("../../Data/subpixeltest.png", image);
    }
} // Namespace SfM::detect