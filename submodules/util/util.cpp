#include "util.hpp"
#include <iostream>
#include <math.h>

namespace SfM::util
{
    Mat4 calculateTransformationMatrixDeg(REAL rotX, REAL rotY, REAL rotZ, Vec3 translation)
    {
        rotX = rotX * M_PI / (REAL)180;
        rotY = rotY * M_PI / (REAL)180;
        rotZ = rotZ * M_PI / (REAL)180;

        return calculateTransformationMatrixRad(rotX, rotY, rotZ, translation);
    }

    Mat4 calculateTransformationMatrixRad(REAL rotX, REAL rotY, REAL rotZ, Vec3 translation)
    {
        Mat3 Rx;
        Rx << (REAL)1, (REAL)0, (REAL)0,
            (REAL)0, std::cos(rotX), -std::sin(rotX),
            (REAL)0, std::sin(rotX), std::cos(rotX);

        Mat3 Ry;
        Ry << std::cos(rotY), (REAL)0, std::sin(rotY),
            (REAL)0, (REAL)1, (REAL)0,
            -std::sin(rotY), (REAL)0, std::cos(rotY);

        Mat3 Rz;
        Rz << std::cos(rotZ), -std::sin(rotZ), (REAL)0,
            std::sin(rotZ), std::cos(rotZ), (REAL)0,
            (REAL)0, (REAL)0, (REAL)1;

        Mat3 R = Rz * Ry * Rx;
        Mat4 res = Mat4::Identity();
        res.block<3, 3>(0, 0) = R;
        res.block<3, 1>(0, 3) = translation;
        return res;
    }

    Vec3rgb getPixelBilinearUchar(const cv::Mat &img, Vec2 pos)
    {
        REAL u = pos[0];
        REAL v = pos[1];
        int x0 = static_cast<int>(std::floor(u));
        int y0 = static_cast<int>(std::floor(v));

        x0 = std::clamp(x0, 0, img.cols - 2);
        y0 = std::clamp(y0, 0, img.rows - 2);

        int x1 = x0 + 1;
        int y1 = y0 + 1;

        REAL dx = u - x0;
        REAL dy = v - y0;
        REAL w00 = (1.0 - dx) * (1.0 - dy);
        REAL w10 = dx * (1.0 - dy);
        REAL w01 = (1.0 - dx) * dy;
        REAL w11 = dx * dy;

        cv::Vec3b p00 = img.at<cv::Vec3b>(y0, x0);
        cv::Vec3b p10 = img.at<cv::Vec3b>(y0, x1);
        cv::Vec3b p01 = img.at<cv::Vec3b>(y1, x0);
        cv::Vec3b p11 = img.at<cv::Vec3b>(y1, x1);

        Vec3rgb result;
        for (int i = 0; i < 3; ++i)
        {
            REAL val = w00 * p00[i] + w10 * p10[i] + w01 * p01[i] + w11 * p11[i];
            result[i] = cv::saturate_cast<uchar>(val);
        }

        // convert from BGR to RGB
        uchar temp = result[0];
        result[0] = result[2];
        result[2] = temp;

        return result;
    }

    template <>
    cv::Mat imageToCvMat<uchar>(const Image<uchar> &image)
    {
        if (image.data == nullptr || image.width <= 0 || image.height <= 0)
        {
            return cv::Mat();
        }
        cv::Mat wrappedMat(image.height, image.width, CV_8UC3, (void *)image.data);
        cv::cvtColor(wrappedMat, wrappedMat, cv::COLOR_RGB2BGR);
        return wrappedMat.clone();
    }

    template <>
    cv::Mat imageToCvMat<float>(const Image<float> &image)
    {
        if (image.data == nullptr || image.width <= 0 || image.height <= 0)
        {
            return cv::Mat();
        }
        std::cout << "IMG to cv mat" << std::endl;
        cv::Mat wrapped(image.height, image.width, CV_32FC1, (void *)image.data);
        return wrapped.clone();
    }

    template <>
    cv::Mat imageToCvMat<double>(const Image<double> &image)
    {
        if (image.data == nullptr || image.width <= 0 || image.height <= 0)
        {
            return cv::Mat();
        }

        cv::Mat wrapped(image.height, image.width, CV_64FC1, (void *)image.data);
        return wrapped.clone();
    }

    bool drawPointsOnImage(const cv::Mat &image,
                           const std::string &outputImagePath,
                           const std::vector<Vec2> &uvPoints,
                           bool drawCircles,
                           int markerSize,
                           cv::Scalar color)
    {
        // Load image (BGR)
        if (image.empty())
            return false;

        const int width = image.cols;
        const int height = image.rows;

        for (const auto &uv : uvPoints)
        {
            int x = static_cast<int>(uv.x());
            int y = static_cast<int>(uv.y());

            // Skip points outside image
            if (x < 0 || x >= width || y < 0 || y >= height)
                continue;

            cv::Point p(x, y);

            if (drawCircles)
            {
                cv::circle(image, p, markerSize, color, 2);
            }
            else
            {
                cv::rectangle(
                    image,
                    cv::Point(x - markerSize, y - markerSize),
                    cv::Point(x + markerSize, y + markerSize),
                    color,
                    2);
            }
        }

        return cv::imwrite(outputImagePath, image);
    }

    void drawCollageWithTracks(const std::vector<cv::Mat> &images,
                               const std::vector<std::vector<Vec2>> &tracks,
                               int startFrame,
                               int endFrame,
                               const std::string &outputPath,
                               int markerSize)
    {
        if (startFrame < 0)
            startFrame = 0;
        if (endFrame >= static_cast<int>(images.size()))
            endFrame = static_cast<int>(images.size()) - 1;
        if (startFrame > endFrame)
            return;

        // Load images and compute total collage size
        int totalWidth = 0;
        int maxHeight = 0;

        for (int i = startFrame; i <= endFrame; ++i)
        {
            const cv::Mat &img = images[i];
            if (img.empty())
                continue;

            totalWidth += img.cols;
            if (img.rows > maxHeight)
                maxHeight = img.rows;
        }

        if (images.empty())
            return;

        // Create the collage canvas
        cv::Mat collage(maxHeight, totalWidth, images[0].type(), cv::Scalar(0, 0, 0));

        // Copy images side by side and store x-offsets
        std::vector<int> xOffsets;
        int currentX = 0;
        for (const auto &img : images)
        {
            img.copyTo(collage(cv::Rect(currentX, 0, img.cols, img.rows)));
            xOffsets.push_back(currentX);
            currentX += img.cols;
        }

        // Draw points and lines
        int numTracks = static_cast<int>(tracks.size());
        int numImages = static_cast<int>(images.size());

        for (int t = 0; t < numTracks; ++t)
        {
            for (int f = 0; f < numImages; ++f)
            {
                const auto &pt = tracks[t][startFrame + f];
                int imgWidth = images[f].cols;
                int imgHeight = images[f].rows;

                //// UV -> pixel coordinates
                // int x = static_cast<int>(pt.x() * imgWidth) + xOffsets[f];
                // int y = static_cast<int>(pt.y() * imgHeight);

                int x = static_cast<int>(pt.x()) + xOffsets[f];
                int y = static_cast<int>(pt.y());
                // Draw the point
                cv::circle(collage, cv::Point(x, y), markerSize, cv::Scalar(0, 255, 0), -1);

                // Draw line to next image if exists
                if (f + 1 < numImages)
                {
                    const auto &nextPt = tracks[t][startFrame + f + 1];
                    int nextX = static_cast<int>(nextPt.x()) + xOffsets[f + 1];
                    int nextY = static_cast<int>(nextPt.y());
                    cv::line(collage, cv::Point(x, y), cv::Point(nextX, nextY), cv::Scalar(0, 0, 255), 2);
                }
            }
        }

        // Save output
        cv::imwrite(outputPath, collage);
    }
} // Namespace SfM::util