#include "blender.hpp"
#include "../util/util.hpp"

namespace SfM::io
{
    void exportTracksForBlender(std::vector<Mat4> &cameraExtrinsics, std::vector<Vec3> &points, std::string path)
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

        // 4. Write Camera Poses (converted to Blender space)
        out << "\n# Camera frames \n";
        for (const auto &pose : cameraExtrinsics)
        {
            // Apply conversion: NewPose = OldPose * T_flip
            SfM::Mat4 blenderPose = util::cvToBlender(pose);
            out << pose.format(CleanFmt) << "\n";
        }

        // 5. Write Points (World coordinates are the same, just the camera interprets them differently)
        out << "\n# Points \n";
        for (const auto &p : points)
        {
            out << p.format(CleanFmt) << "\n";
        }

        out.close();
        std::cout << "Exported " << cameraExtrinsics.size() << " frames and "
                  << points.size() << " points to " << path << std::endl;
    }
} // Namespace SfM::io