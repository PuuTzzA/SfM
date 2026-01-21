#pragma once
#include "../SfM.hpp"
#include <string>
#include <opencv2/opencv.hpp>
#include <turbojpeg.h>

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
     * @brief Converts a struct Image to a cv::Mat.
     * @param image Input image
     * @return cv::Mat of the image (rgb if T == ucar, monochrome if T == REAL)
     */
    template <typename T>
    cv::Mat imageToCvMat(const Image<T> &image);

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
     * @brief Draws points on an image using UV coordinates and saves the result.
     *
     * @param inputImagePath Path to the input image.
     * @param outputImagePath Path to the output image.
     * @param uvPoints Vector of points in normalized UV coordinates [0,1].
     * @param drawCircles If true, points are drawn as circles; otherwise as squares.
     * @param markerSize Radius/half-size of the point markers in pixels.
     * @param color Color of the drawn markers.
     * @return True if the image was successfully saved, false on failure.
     */
    bool drawPointsOnImage(const std::string &inputImagePath,
                           const std::string &outputImagePath,
                           const std::vector<Vec2> &uvPoints,
                           bool drawCircles = true,
                           int markerSize = 5,
                           cv::Scalar color = cv::Scalar(0, 0, 255));

    /**
     * @brief Visualizes tracked points over a range of frames.
     *
     * @param imagePaths Vector of input images.
     * @param tracks Vector of tracked points.
     * @param startFrame use frames from startFrame ...
     * @param endFrame ... to endFrame
     * @param outputPath Path to output image.
     * @param markerSize Size of the markers.
     * @return True if the image was successfully saved, false on failure.
     */
    void drawCollageWithTracks(const std::vector<std::string> &imagePaths,
                               const std::vector<std::vector<Vec2>> &tracks,
                               int startFrame,
                               int endFrame,
                               const std::string &outputPath,
                               int markerSize = 5);

    /**
     * @brief Loads all images inside a directory. Only files with the extension ".png", ".jpeg" or ".jpg" will be loaded.
     *
     * @param dir Path to the directory containing the images.
     * @return An std::vector of all the loaded images.
     */
    std::vector<cv::Mat> loadImages(std::string dir);
} // namespace SfM::io
