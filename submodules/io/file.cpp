#include "file.hpp"
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>

namespace SfM::io
{
    std::string getExtension(const std::string &path)
    {
        size_t dotPos = path.rfind('.');
        if (dotPos == std::string::npos)
            return "";
        std::string ext = path.substr(dotPos + 1);
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        return ext;
    }

    Image<uchar> loadJpg(const std::string &path, int flags)
    {
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        if (!file)
        {
            std::cerr << "Failed to open file: " << path << std::endl;
            return Image<uchar>(0, 0);
        }
        std::streamsize size = file.tellg();
        file.seekg(0, std::ios::beg);
        uchar *jpegBuffer = new uchar[size];
        if (!file.read((char *)jpegBuffer, size))
        {
            return Image<uchar>(0, 0);
        }

        // Initialize Decompressor
        tjhandle decompressor = tjInitDecompress();
        int width, height, subsamp, colorspace;

        // Read Header
        if (tjDecompressHeader3(decompressor, jpegBuffer, size, &width, &height, &subsamp, &colorspace) < 0)
        {
            tjDestroy(decompressor);
            return Image<uchar>(0, 0);
        }

        // Allocate RGB Vector (3 bytes per pixel)
        Image<uchar> retImg(width, height, 3);

        // Decompress directly to vector
        // TJPF_RGB ensures standard RGB byte order
        if (tjDecompress2(decompressor, jpegBuffer, size, retImg.data, width,
                          0, height, TJPF_RGB, flags) < 0)
        {
            tjDestroy(decompressor);
            return Image<uchar>(0, 0);
        }

        tjDestroy(decompressor);
        delete[] jpegBuffer;
        return retImg;
    }

    Image<uchar> loadImage(const std::string &path, int turboJpegFlags)
    {
        std::string ext = getExtension(path);
        if (ext == "jpg" || ext == "jpeg") // INFO: turbojpeg is consistantly 30-60ms faster than OpenCv (1920x1080)
        {
            std::cout << "Loading image with turbojpeg" << std::endl;
            return loadJpg(path, turboJpegFlags);
        }
        else
        {
            std::cout << "Loading image with OpenCv" << std::endl; // INFO: loadPNG was slower than OpenCV, therefore I removed it again
            cv::Mat img = cv::imread(path, cv::IMREAD_COLOR);

            if (img.empty())
            {
                std::cerr << "OpenCV failed to load: " << path << std::endl;
                return Image<uchar>(0, 0);
            }

            cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
            Image<uchar> result(img.cols, img.rows, 3);

            size_t totalBytes = img.cols * img.rows * 3 * sizeof(uchar);

            if (img.isContinuous())
            {
                std::memcpy(result.data, img.data, totalBytes);
            }
            else
            {
                for (int y = 0; y < img.rows; ++y)
                {
                    std::memcpy(result.data + y * img.cols * 3, img.ptr<uchar>(y), img.cols * 3 * sizeof(uchar));
                }
            }

            return result;
        }
    }

    template <>
    cv::Mat imageToCvMat<uchar>(const Image<uchar> &image)
    {
        if (image.data == nullptr || image.width <= 0 || image.height <= 0)
        {
            return cv::Mat();
        }
        std::cout << "moin: " << image.width << ", " << image.height << ", " << std::endl;
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

    std::vector<std::vector<Vec2>> loadTrackedPoints(const std::string &path)
    {
        std::ifstream file(path);
        if (!file.is_open())
            return {};

        std::vector<std::vector<Vec2>> result;

        int trackIndex, frame;
        float x, y;

        while (file >> trackIndex >> frame >> x >> y)
        {
            if (trackIndex >= static_cast<int>(result.size()))
            {
                result.resize(trackIndex + 1);
            }

            result[trackIndex].emplace_back(x, y);
        }

        return result;
    }

    bool drawPointsOnImage(const std::string &inputImagePath,
                           const std::string &outputImagePath,
                           const std::vector<Vec2> &uvPoints,
                           bool drawCircles,
                           int markerSize,
                           cv::Scalar color)
    {
        // Load image (BGR)
        cv::Mat image = cv::imread(inputImagePath, cv::IMREAD_COLOR);
        if (image.empty())
            return false;

        const int width = image.cols;
        const int height = image.rows;

        for (const auto &uv : uvPoints)
        {
            // Convert normalized UV [0,1] to pixel coordinates
            int x = static_cast<int>(uv.x() * width);
            int y = static_cast<int>(uv.y() * height);

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

    void drawCollageWithTracks(const std::vector<std::string> &imagePaths,
                               const std::vector<std::vector<Vec2>> &tracks,
                               int startFrame,
                               int endFrame,
                               const std::string &outputPath,
                               int markerSize)
    {
        if (startFrame < 0)
            startFrame = 0;
        if (endFrame >= static_cast<int>(imagePaths.size()))
            endFrame = static_cast<int>(imagePaths.size()) - 1;
        if (startFrame > endFrame)
            return;

        // Load images and compute total collage size
        std::vector<cv::Mat> images;
        int totalWidth = 0;
        int maxHeight = 0;

        for (int i = startFrame; i <= endFrame; ++i)
        {
            cv::Mat img = cv::imread(imagePaths[i], cv::IMREAD_COLOR);
            if (img.empty())
                continue;

            images.push_back(img);
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

                // UV -> pixel coordinates
                int x = static_cast<int>(pt.x() * imgWidth) + xOffsets[f];
                int y = static_cast<int>(pt.y() * imgHeight);

                // Draw the point
                cv::circle(collage, cv::Point(x, y), markerSize, cv::Scalar(0, 255, 0), -1);

                // Draw line to next image if exists
                if (f + 1 < numImages)
                {
                    const auto &nextPt = tracks[t][startFrame + f + 1];
                    int nextX = static_cast<int>(nextPt.x() * images[f + 1].cols) + xOffsets[f + 1];
                    int nextY = static_cast<int>(nextPt.y() * images[f + 1].rows);
                    cv::line(collage, cv::Point(x, y), cv::Point(nextX, nextY), cv::Scalar(0, 0, 255), 2);
                }
            }
        }

        // Save output
        cv::imwrite(outputPath, collage);
    }
} // Namespace SfM::io
