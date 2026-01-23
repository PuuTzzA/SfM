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

    /**
     * @brief Undistorts a given image.
     * @param image The image that should be undistorted.
     * @param calibration The calibration of the camera that created the image.
     * @return The undistorted image.
     */
    cv::Mat undistort(const cv::Mat& image, const CameraCalibration& calibration);

    /**
     * @brief Undistorts a given image.
     * @param image The image that should be undistorted.
     * @param calibration The calibration of the camera that created the image.
     * @return The undistorted image.
     */
    template<class T> Image<T> undistort(const Image<T>& image, const CameraCalibration& calibration) {
        
    }

} // namespace Sfm::calibrate