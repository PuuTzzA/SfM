#include "calibrate.hpp"

namespace SfM::calibrate {

    CameraCalibration calibrateCamera(const std::vector<cv::Mat>& images, cv::Size boardSize) {

        float squareSize = 1.0f;

        std::cout << "Setting object points\n";
        std::vector<cv::Point3f> objectPoints{};

        for(uint32_t i = 0; i < boardSize.height; i++) {
            for(uint32_t j = 0; j < boardSize.width; j++) {
                objectPoints.emplace_back(j * squareSize, i * squareSize, 0.0f);
            }
        }

        std::vector<std::vector<cv::Point3f>> objectPointsList{};
        std::vector<std::vector<cv::Point2f>> imagePointsList{};

        cv::Size imageSize;

        for(const auto& image : images) {
            imageSize = image.size();

            std::vector<cv::Point2f> corners;

            std::cout << "Finding chessboard corners, image size: " << image.rows << ", " << image.cols << "\n";
            bool found = cv::findChessboardCorners(image, boardSize, corners, cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE);
            std::cout << "Found: " << found << std::endl;
            if(!found)
                continue;

            cv::Mat grayscaleImage;
            cv::cvtColor(image, grayscaleImage, cv::COLOR_BGR2GRAY);

            auto criteria = cv::TermCriteria{cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.001};

            cv::cornerSubPix(grayscaleImage, corners, cv::Size{11, 11}, cv::Size{-1, -1}, criteria);

            imagePointsList.push_back(corners);
            objectPointsList.push_back(objectPoints);
        }

        CameraCalibration calibration{};
        std::vector<cv::Mat> rvecs, tvecs;

        double rms = cv::calibrateCamera(
            objectPointsList,
            imagePointsList,
            imageSize,
            calibration.matrix,
            calibration.distortionCoeffs,
            rvecs,
            tvecs
        );

        return calibration;
    }

    cv::Mat undistort(const cv::Mat& image, const CameraCalibration& calibration) {
        cv::Mat undistorted;
        cv::undistort(image, undistorted, calibration.matrix, calibration.distortionCoeffs);
        
        return undistorted;
    }

} // namespace SfM::calibrate