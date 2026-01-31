#include "file.hpp"
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <filesystem>

#include "../../external/nlohmann/json.hpp"

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

    std::vector<cv::Mat> loadImages(const std::string &dir)
    {
        std::vector<cv::Mat> images{};

        std::filesystem::path dir_path{dir};

        auto validExtension = [](const std::string &ext)
        {
            return ext == ".png" || ext == ".jpeg" || ext == ".jpg";
        };

        for (const auto &dir_entry : std::filesystem::directory_iterator{dir_path})
        {
            auto entry_path = dir_entry.path();

            if (dir_entry.is_directory() || !validExtension(entry_path.extension().string()))
                continue;

            std::cout << "Loading image file '" << entry_path << "'" << std::endl;
            cv::Mat image = cv::imread(entry_path.string(), cv::IMREAD_COLOR);

            if (image.empty())
            {
                std::cerr << "OpenCV failed to load file '" << entry_path << "'. Ignored!" << std::endl;
                continue;
            }

            images.push_back(image);
        }

        return images;
    }

    void storeCalibration(const std::string &path, const SfM::calibrate::CameraCalibration &calibration)
    {
        using namespace nlohmann;

        json matrix = json::array();
        json distCoeffs = json::array();

        std::cout << calibration.matrix.type() << std::endl;

        for (int i = 0; i < 3; i++)
        {
            auto row = json::array();
            for (int j = 0; j < 3; j++)
            {
                row.push_back(calibration.matrix.at<double>(i, j));
            }
            matrix.push_back(row);
        }

        for (int i = 0; i < 5; i++)
        {
            distCoeffs.push_back(calibration.distortionCoeffs.at<double>(i));
        }

        json data{};

        data["matrix"] = matrix;
        data["distortion"] = distCoeffs;

        std::ofstream file(path);

        if (!file)
        {
            std::cerr << "Failed to open file: " << path << std::endl;
            return;
        }
        file << data.dump(4);
        file.close();
    }

    SfM::calibrate::CameraCalibration loadCalibration(const std::string &path)
    {
        using namespace nlohmann;

        std::ifstream file(path);

        if (!file)
        {
            std::cerr << "Failed to open file: " << path << std::endl;
            return {};
        }

        json data = json::parse(file);

        const json &matrix = data["matrix"];
        const json &distCoeffs = data["distortion"];

        cv::Matx33d cvMatrix{};
        cv::Vec<double, 5> cvDistCoeffs{};
        Mat3 K;

        for (int i = 0; i < 5; i++)
        {
            cvDistCoeffs[i] = distCoeffs[i];
        }

        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                K(i, j) = matrix[i][j];
                cvMatrix(i, j) = matrix[i][j];
            }
        }

        return calibrate::CameraCalibration{K, cv::Mat{cvMatrix}, cv::Mat{cvDistCoeffs}};
    }
} // Namespace SfM::io
