#pragma once
#include "../SfM.hpp"
#include <opencv2/opencv.hpp>

namespace SfM::detect
{
    void harrisCornerDetectionOpenCv(cv::Mat &image);
    
    void harrisCornerDetectionSubPixelOpenCv(cv::Mat &image);
} // Namespace SfM::detect