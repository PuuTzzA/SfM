#pragma once
#include <Eigen/Core>
#include <vector>

namespace SfM
{
    using REAL = float;

    using Vec2 = Eigen::Matrix<REAL, 2, 1>;
    using Vec3 = Eigen::Matrix<REAL, 3, 1>;
    using Vec4 = Eigen::Matrix<REAL, 4, 1>;
    using VecX = Eigen::Matrix<REAL, Eigen::Dynamic, 1>;

    using Mat2 = Eigen::Matrix<REAL, 2, 2>;
    using Mat3 = Eigen::Matrix<REAL, 3, 3>;
    using Mat4 = Eigen::Matrix<REAL, 4, 4>;
    using MatX = Eigen::Matrix<REAL, Eigen::Dynamic, Eigen::Dynamic>;

    struct SfMResult
    {
        Mat4 pose;                // Pose of Camera 2
        std::vector<Vec3> points; // Reconstructed 3D points
    };
} // Namespace SfM