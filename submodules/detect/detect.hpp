#pragma once
#include "../SfM.hpp"
#include <opencv2/opencv.hpp>

namespace SfM::detect
{
    /**
     * @brief NOT FINISHED, DUE TO IT NOT BEING USEFUL IN SIFT. Uses the Harris Corner Detector to find conrners. Uses an iterative approach to refine the corners to subpixel accuracy.
     * @param image Image<uchar>
     * @param blockSize Size of the neighbourhood considered for corner detection
     * @param maxIter Maximal interaion count in corner refinement
     * @param maxDelta Minimal change between iterations in corner refinement
     * @returns vector<Vec2> of corners in pixel coordinates
     */
    std::vector<Vec2> harrisCornerDetection(const Image<uchar> &image, const int blockSize, const int maxIter = 100, const REAL maxDelta = 0.001);

    /**
     * @brief Converts a rgb image to REAL.
     * @param image rgb image
     * @return image as REAL
     */
    template<typename T>
    Image<T> rgbToREAL(const Image<uchar> &image);

    void harrisCornerDetectionOpenCv(cv::Mat &image);

    std::vector<Vec2> harrisCornerDetectionSubPixelOpenCv(cv::Mat &image);

    void siftCv(cv::Mat& image);
} // Namespace SfM::detect