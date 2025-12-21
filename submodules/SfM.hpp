#pragma once
#include <Eigen/Core>
#include <vector>

namespace SfM
{
    using REAL = double;

    using Vec2 = Eigen::Matrix<REAL, 2, 1>;
    using Vec3 = Eigen::Matrix<REAL, 3, 1>;
    using Vec4 = Eigen::Matrix<REAL, 4, 1>;
    using VecX = Eigen::Matrix<REAL, Eigen::Dynamic, 1>;

    using Mat2 = Eigen::Matrix<REAL, 2, 2>;
    using Mat3 = Eigen::Matrix<REAL, 3, 3>;
    using Mat4 = Eigen::Matrix<REAL, 4, 4>;
    using MatX = Eigen::Matrix<REAL, Eigen::Dynamic, Eigen::Dynamic>;

    /**
     * @brief An observation represents one 2D measurement
     */
    struct Observation
    {
        int frameId; // which image/frame
        Vec2 point;  // 2D measurement in that frame
    };

    /**
     * @brief A Track a track represents one physical 3D point, e.g.:
     * 
     * Track #42:
     *  frame 0 → (120.3, 85.1)
     *  frame 1 → (118.9, 84.7)
     *  frame 3 → (115.2, 83.9)
     */
    struct Track
    {
        int id;                                // stable identity
        std::vector<Observation> observations; // all frames where it exists
    };

    struct SfMResult
    {
        Mat4 pose;                // Pose of Camera 2
        std::vector<Vec3> points; // Reconstructed 3D points
    };
} // Namespace SfM