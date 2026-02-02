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

    std::vector<cv::Mat> loadImages(const std::string &dir, std::vector<double> *timestamps, uint32_t limit)
    {
        std::vector<cv::Mat> images{};
        if (timestamps)
            *timestamps = std::vector<double>{};

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
            if (timestamps)
            {
                try
                {
                    timestamps->push_back(std::stod(entry_path.stem().string()));
                }
                catch (std::exception e)
                {
                    timestamps->push_back(NAN);
                }
            }

            if (images.size() == limit)
            {
                break;
            }
        }

        return images;
    }

    void storeImages(Scene &scene, const std::string &path, const std::string &name, const std::string &extension)
    {
        namespace fs = std::filesystem;

        const std::vector<cv::Mat> &images = scene.getImages();

        if (!path.empty() && !fs::exists(path))
        {
            fs::create_directories(path);
        }

        for (size_t i = 0; i < images.size(); ++i)
        {
            std::ostringstream oss;
            oss << name << std::setw(4) << std::setfill('0') << i << "." << extension;

            fs::path fullPath = fs::path(path) / oss.str();

            if (!images[i].empty())
            {
                cv::imwrite(fullPath.string(), images[i]);
            }
        }
    }

    void storeCalibration(const std::string &path, const SfM::calibrate::CameraCalibration &calibration)
    {
        using namespace nlohmann;

        json matrix = json::array();
        json distCoeffs = json::array();

        // std::cout << calibration.matrix.type() << std::endl;

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

        std::cout << "Exported Calibratioin to " << path << std::endl;
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

    void exportTrack(std::vector<Mat4> &extrinsics, const std::vector<double> &timestamps, const std::string &path, const std::string &ground_truth_path)
    {
        std::ifstream ground_truth(ground_truth_path);

        double tx, ty, tz, qx, qy, qz, qw;

        std::string line;

        std::string prev_line = "";

        auto find_timestamp = [&](double t)
        {
            while (std::getline(ground_truth, line))
            {
                std::istringstream iss(line);
                double timestamp;

                iss >> timestamp;

                if (timestamp > t)
                {
                    std::istringstream prev_iss(prev_line);

                    double prev_timestamp;

                    prev_iss >> prev_timestamp;

                    std::istringstream best_iss;

                    if (abs(timestamp - t) < abs(prev_timestamp - t))
                    {
                        best_iss = std::move(iss);
                    }
                    else
                    {
                        best_iss = std::move(prev_iss);
                    }

                    best_iss >> tx >> ty >> tz >> qx >> qy >> qz >> qw;

                    std::cout << tx << " " << ty << " " << tz << std::endl;

                    break;
                }

                prev_line = line;
            }
        };

        find_timestamp(timestamps[0]);

        Mat4 gMat = Mat4::Identity();

        Eigen::Quaternion<REAL> gq = Eigen::Quaternion{qw, qx, qy, qz};

        Vec3 gtStart{tx, ty, tz};

        gMat.block<3, 3>(0, 0) = Mat3(gq);
        gMat.block<3, 1>(0, 3) = Vec3{tx, ty, tz};

        ground_truth.clear();
        ground_truth.seekg(0, std::ios::beg);

        find_timestamp(timestamps.back());

        Vec3 gtEnd{tx, ty, tz};

        double scale = (gtEnd - gtStart).norm() / (extrinsics.back().block<3, 1>(0, 3) - extrinsics[0].block<3, 1>(0, 3)).norm();

        std::cout << scale << std::endl;

        for (auto &pose : extrinsics)
        {
            pose.block<3, 1>(0, 3) *= scale;
        }

        Mat4 universal_transform = gMat * extrinsics[0].inverse();
        std::ofstream file(path);

        if (!file)
        {
            std::cerr << "Could not open file '" << path << "'\n";
        }

        for (uint32_t i = 0; i < extrinsics.size(); i++)
        {
            auto timestamp = timestamps[i];
            auto pose = universal_transform * extrinsics[i];

            Vec3 t = pose.block<3, 1>(0, 3);
            Mat3 rotationMat = pose.block<3, 3>(0, 0);
            Eigen::Quaternion<REAL> q{rotationMat};

            file << std::fixed << std::setprecision(4) << timestamp << " " << t.x() << " " << t.y() << " " << t.z() << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << std::endl;
        }

        file.close();
    }
} // Namespace SfM::io
