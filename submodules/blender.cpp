#include <iostream>
#include <Eigen/Core>
#include "solve.h"

namespace SfM::Blender
{
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
}