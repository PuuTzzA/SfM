#pragma once

#include "../SfM.hpp"

#include <vector>
#include <opencv2/opencv.hpp>

namespace SfM::calibrate {

    struct CameraCalibration {
        cv::Mat matrix;
        cv::Mat distortionCoeffs;
    };

    /**
     * @brief Returns the camera calibration calculated from an array of images.
     * @param images The images containing the chessboard pattern from different angles.
     * @return CameraCalibration which contains the calibration parameters.
     */
    CameraCalibration calibrateCamera(const std::vector<cv::Mat>& images);

} // namespace Sfm::calibrate