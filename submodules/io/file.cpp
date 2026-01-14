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

    std::vector<uchar> loadJpg(const std::string &path, int flags)
    {
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        if (!file)
        {
            std::cerr << "Failed to open file: " << path << std::endl;
            return {};
        }
        std::streamsize size = file.tellg();
        file.seekg(0, std::ios::beg);
        std::vector<uchar> jpegBuffer(size);
        if (!file.read((char *)jpegBuffer.data(), size))
            return {};

        // Initialize Decompressor
        tjhandle decompressor = tjInitDecompress();
        int width, height, subsamp, colorspace;

        // Read Header
        if (tjDecompressHeader3(decompressor, jpegBuffer.data(), size, &width, &height, &subsamp, &colorspace) < 0)
        {
            tjDestroy(decompressor);
            return {};
        }

        // Allocate RGB Vector (3 bytes per pixel)
        std::vector<uchar> pixels(width * height * 3);

        // Decompress directly to vector
        // TJPF_RGB ensures standard RGB byte order
        if (tjDecompress2(decompressor, jpegBuffer.data(), size, pixels.data(), width,
                          0, height, TJPF_RGB, flags) < 0)
        {
            tjDestroy(decompressor);
            return {};
        }

        tjDestroy(decompressor);
        return pixels;
    }

    std::vector<uchar> loadImage(const std::string &path, int turboJpegFlags)
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
            cv::cvtColor(img, img, cv::COLOR_BGR2RGB); // Convert BGR to RGB
            if (img.empty())
            {
                std::cerr << "OpenCV failed to load: " << path << std::endl;
                return {};
            }
            return cvMatToVector(img);
        }
    }

    std::vector<uchar> cvMatToVector(cv::Mat &mat)
    {
        if (mat.empty())
        {
            return std::vector<uchar>();
        }

        // 2. Handle memory continuity
        // If the matrix is a Region of Interest (ROI) or has padding, the memory
        // is not continuous. We must make it continuous to copy it into a flat vector.
        // Since 'mat' is an rvalue reference (&&), we can modify it or clone it
        // internally without affecting the caller's "original" variable (which is dying anyway).
        if (!mat.isContinuous())
        {
            mat = mat.clone();
        }

        // 3. Calculate total bytes
        // total() returns the number of pixels.
        // elemSize() returns the number of bytes per pixel (e.g., 3 for CV_8UC3, 4 for float).
        size_t sizeInBytes = mat.total() * mat.elemSize();

        // 4. Copy data to vector
        // We use the pointer to the start of the data and the calculated size.
        // std::vector copies the data into its own managed memory.
        return std::vector<uchar>(mat.data, mat.data + sizeInBytes);
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
