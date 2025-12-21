#pragma once
#include "../SfM.hpp"
#include <iostream>
#include <fstream>
#include <string>

namespace SfM::io
{
    /**
     * @brief Exports tracks to a file that can be imported into blender. NOTE: All cameraExtrinsics, points given in CV coordinates (look: +Z, Y: down)
     *
     * @param cameraExtrinsics Vector of cameraExtrinsics
     * @param points Vector of 3d points
     * @param path String to output path
     */
    void exportTracksForBlender(std::vector<Mat4> &cameraExtrinsics, std::vector<Vec3> &points, std::string path);

    void printForBlender(const SfMResult &res);
} // Namespace SfM::io