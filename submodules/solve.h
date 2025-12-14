#pragma once
#include <Eigen/Core>
#include <vector>

namespace SfM
{
    struct SfMResult
    {
        Eigen::Matrix4f pose;                // Pose of Camera 2
        std::vector<Eigen::Vector3f> points; // Reconstructed 3D points
    };
} // Namespace SfM

namespace SfM::Solve
{
    SfMResult eightPointAlgorithm(std::vector<std::vector<Eigen::Vector2f>> tracks);
} // Namespace SfM::Solve