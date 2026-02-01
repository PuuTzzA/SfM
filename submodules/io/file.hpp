#pragma once
#include "../SfM.hpp"
#include "../scene.hpp"
#include "../calibrate/calibrate.hpp"

#include <string>
#include <opencv2/opencv.hpp>
#include <turbojpeg.h>
#include <vector>

namespace SfM::io
{
    /**
     * @brief Loads an image from disk into a std::vector.
     * If the image is a .jpg (.jpeg) then use turbojpeg for maximal performance.
     * The file is stored like this in memory: [r, g, b, r, g, b, r, g, b, ...] from top left, row by row
     *
     * @param path Path to the image
     * @param turboJpegFlags (default TJFLAG_ACCURATEDCT) Flags for turboJpeg (no big speed difference between TJFLAG_ACCURATEDCT and TJFLAG_FASTDCT)
     * @return Image wrapper struct {data, width, height}
     */
    Image<uchar> loadImage(const std::string &path, int turboJpegFlags = TJFLAG_ACCURATEDCT);

    /**
     * @brief Reads tracked points from a file.
     *
     * @param inputImagePath Path to the input file. Expects the following format:
     * - #id #frame x y
     * - #id #frame x y
     * ...
     * @return Vector of tracked points. result[0] = first track, result[0][0] = first track, first frame
     */
    std::vector<std::vector<Vec2>> loadTrackedPoints(const std::string &path);

    /**
     * @brief Loads all images inside a directory. Only files with the extension ".png", ".jpeg" or ".jpg" will be loaded.
     *
     * @param dir Path to the directory containing the images.
     * @return An std::vector of all the loaded images.
     */
    std::vector<cv::Mat> loadImages(const std::string& dir, std::vector<double>* timestamps = nullptr, uint32_t limit = 0);

    /**
     * @brief Stores a camera calibration in a file.
     *
     * @param path The path of the file.
     * @param calibration The camera calibration to be stored.
     */
    void storeCalibration(const std::string& path, const SfM::calibrate::CameraCalibration& calibration);

    /**
     * @brief Loads a camera calibration from a file.
     *
     * @param path The path of the file.
     * @return The camera calibration stored in the file.
     */
    SfM::calibrate::CameraCalibration loadCalibration(const std::string& path);

    /**
     * @brief Exports the track in a format that allows for evaluation.
     *
     * @param scene The scene containing the track.
     * @param timestamps The timestamps of each camera pose in the track.
     * @param path The path of the file.
     */
    void exportTrack(std::vector<Mat4>& extrinsics, const std::vector<double>& timestamps, const std::string& path);

} // namespace SfM::io
