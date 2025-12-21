#include "blender.hpp"

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

        // This flips Y and Z axes to match Blender's camera convention (Look -Z, Up +Y)
        SfM::Mat4 cvToBlender = SfM::Mat4::Identity();
        cvToBlender(1, 1) = -1.0;
        cvToBlender(2, 2) = -1.0;

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
            SfM::Mat4 blenderPose = pose * cvToBlender;
            out << blenderPose.format(CleanFmt) << "\n";
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

    void printForBlender(const SfMResult &res)
    {
        std::cout << "\n# --- COPY TO BLENDER SCRIPT ---" << std::endl;

        // Print Matrix
        std::cout << "cv_matrix = [" << std::endl;
        for (int i = 0; i < 4; i++)
        {
            std::cout << "    [";
            for (int j = 0; j < 4; j++)
                std::cout << res.pose(i, j) << (j < 3 ? ", " : "");
            std::cout << "]," << std::endl;
        }
        std::cout << "]" << std::endl;

        // Print Points
        std::cout << "points_data = [" << std::endl;
        for (const auto &p : res.points)
        {
            std::cout << "    (" << p.x() << ", " << p.y() << ", " << p.z() << ")," << std::endl;
        }
        std::cout << "]" << std::endl;
        std::cout << "# ------------------------------" << std::endl;
    }
} // Namespace SfM::io