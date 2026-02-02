#include "blender.hpp"
#include "../util/util.hpp"
#include "../../external/nlohmann/json.hpp"

namespace SfM::io
{
    void exportTracksForBlender(int width,
                                int height,
                                Mat3 K,
                                std::vector<Mat4> &cameraExtrinsics,
                                std::vector<Vec3> &points,
                                std::vector<Vec3rgb> &colors,
                                std::string path,
                                std::string pathToImages)
    {
        using json = nlohmann::ordered_json;

        // General information
        json jsonFile{};
        jsonFile["width"] = width;
        jsonFile["height"] = height;

        if (pathToImages != "")
        {
            jsonFile["pathToImages"] = pathToImages;
        }

        // Camera information
        json intrinsics = json::array();
        for (int i = 0; i < K.rows(); ++i)
        {
            for (int j = 0; j < K.cols(); ++j)
            {
                intrinsics.push_back(K(i, j));
            }
        }
        jsonFile["K"] = intrinsics;

        // Scene information
        json extrinsicsArray = json::array();
        for (const auto &pose : cameraExtrinsics)
        {
            json extrinsics = json::array();
            Mat4 poseWorld = util::cvCameraToBlender(pose);
            for (int i = 0; i < poseWorld.rows(); ++i)
            {
                for (int j = 0; j < poseWorld.cols(); ++j)
                {
                    extrinsics.push_back(poseWorld(i, j));
                }
            }
            extrinsicsArray.push_back(extrinsics);
        }
        jsonFile["extrinsics"] = extrinsicsArray;

        json pointArray = json::array();
        for (const auto &p : points)
        {
            json point = json::array();
            Vec3 pWorld = util::blendCvMat3() * p;
            point.push_back(pWorld[0]);
            point.push_back(pWorld[1]);
            point.push_back(pWorld[2]);
            pointArray.push_back(point);
        }
        jsonFile["points"] = pointArray;

        if (colors.size() == points.size()) // it can be that no color information was provided
        {
            json colArray = json::array();
            for (const auto &c : colors)
            {
                json col = json::array();
                col.push_back(c[0]);
                col.push_back(c[1]);
                col.push_back(c[2]);
                colArray.push_back(col);
            }
            jsonFile["colors"] = colArray;
        }

        // Save to file
        std::ofstream out(path, std::ios::trunc); // trunc ensures overwrite
        if (!out.is_open())
        {
            std::cerr << "Error: Could not open file " << path << " for writing." << std::endl;
            return;
        }

        out << jsonFile.dump(4);
        out.close();
        std::cout << "Exported " << cameraExtrinsics.size() << " frames and "
                  << points.size() << " points to " << path << std::endl;
    };

    void exportSceneForBlender(Scene &scene, std::string path, std::string pathToImages)
    {
        exportTracksForBlender(scene.getImages()[0].cols,
                               scene.getImages()[0].rows,
                               scene.getK(),
                               scene.getExtrinsics(),
                               scene.get3dPointsFilterd(),
                               scene.getColors(),
                               path,
                               pathToImages);
    }

    void exportTracksForBlenderOld(std::vector<Mat4> &cameraExtrinsics, std::vector<Vec3> &points, std::string path, std::string pathToImages)
    {
        std::ofstream out(path, std::ios::trunc); // trunc ensures overwrite
        if (!out.is_open())
        {
            std::cerr << "Error: Could not open file " << path << " for writing." << std::endl;
            return;
        }

        // 2. Define a clean output format for Eigen (single line per matrix/vector)
        Eigen::IOFormat CleanFmt(Eigen::FullPrecision, Eigen::DontAlignCols, " ", " ", "", "", "", "");

        // 3. Write Header: [NumCameras] [NumPoints]
        out << "# Number of camera frames and number of points \n";
        out << cameraExtrinsics.size() << " " << points.size() << "\n";

        std::cout << "Moin inside" << std::endl;
        if (!pathToImages.empty())
        {
            out << "\n# Path to images \n";
            out << pathToImages << "\n";
        }

        // 4. Write Camera Poses (converted to Blender space)
        out << "\n# Camera frames \n";
        for (const auto &pose : cameraExtrinsics)
        {
            out << util::cvCameraToBlender(pose).format(CleanFmt) << "\n";
        }

        // 5. Write Points (World coordinates are the same, just the camera interprets them differently)
        out << "\n# Points \n";
        for (const auto &p : points)
        {
            out << (util::blendCvMat3() * p).format(CleanFmt) << "\n";
        }

        out.close();
        std::cout << "Exported " << cameraExtrinsics.size() << " frames and "
                  << points.size() << " points to " << path << std::endl;
    }
} // Namespace SfM::io