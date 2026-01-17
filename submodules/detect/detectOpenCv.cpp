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

    std::vector<Vec2> harrisCornerDetectionSubPixelOpenCv(cv::Mat &image)
    {
        std::vector<Vec2> eigenCorners;

        // 1. Convert to Grayscale
        cv::Mat gray;
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

        // 2. Prepare for Harris (Float input is standard, though 8U works)
        cv::Mat grayFloat;
        gray.convertTo(grayFloat, CV_32FC1);

        // 3. Harris Corner Detection
        cv::Mat dst;
        cv::cornerHarris(grayFloat, dst, 2, 3, 0.04);
        cv::dilate(dst, dst, cv::Mat());

        // 4. Thresholding
        double maxVal;
        cv::minMaxLoc(dst, nullptr, &maxVal);

        cv::Mat dstThresh;
        cv::threshold(dst, dstThresh, 0.01 * maxVal, 255, cv::THRESH_BINARY);
        dstThresh.convertTo(dstThresh, CV_8UC1);

        // 5. Find Centroids (Initial guesses)
        cv::Mat labels, stats, centroids;
        cv::connectedComponentsWithStats(dstThresh, labels, stats, centroids);

        // Convert centroids to a vector of Point2f for refinement
        // Note: We start at i = 1 to skip the background (label 0)
        std::vector<cv::Point2f> cvCorners;
        std::vector<cv::Point> initialCentroidsInt; // For visualization only

        for (int i = 1; i < centroids.rows; ++i)
        {
            double x = centroids.at<double>(i, 0);
            double y = centroids.at<double>(i, 1);

            cvCorners.push_back(cv::Point2f(static_cast<float>(x), static_cast<float>(y)));
            initialCentroidsInt.push_back(cv::Point(static_cast<int>(x), static_cast<int>(y)));
        }

        // 6. SubPixel Refinement
        if (!cvCorners.empty())
        {
            cv::TermCriteria criteria(
                cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER,
                100,
                0.001);

            // This modifies cvCorners in-place with refined coordinates
            cv::cornerSubPix(gray, cvCorners, cv::Size(5, 5), cv::Size(-1, -1), criteria);
        }

        // 7. Convert to Eigen and Draw Results
        for (size_t i = 0; i < cvCorners.size(); ++i)
        {
            // Convert OpenCV point to Eigen Vec2
            eigenCorners.push_back(Vec2(cvCorners[i].x, cvCorners[i].y));

            // --- Visualization Logic ---
            // Original integer centroid (Red)
            cv::Point c0 = initialCentroidsInt[i];

            // Refined subpixel corner (Green)
            // We cast to int for pixel access, but for accuracy, drawing functions like cv::circle are better
            cv::Point c1(static_cast<int>(cvCorners[i].x), static_cast<int>(cvCorners[i].y));

            // Bounds check to prevent crashes if corners are on the edge
            if (c0.x >= 0 && c0.x < image.cols && c0.y >= 0 && c0.y < image.rows)
                image.at<cv::Vec3b>(c0) = cv::Vec3b(0, 0, 255);

            if (c1.x >= 0 && c1.x < image.cols && c1.y >= 0 && c1.y < image.rows)
                image.at<cv::Vec3b>(c1) = cv::Vec3b(0, 255, 0);
        }

        cv::imwrite("../../Data/subpixeltest.png", image);

        return eigenCorners;
    }

} // Namespace SfM::detect