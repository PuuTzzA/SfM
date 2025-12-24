#pragma once
#include <Eigen/Dense>
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

    constexpr REAL EPSILON = static_cast<REAL>(1e-6);

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

    /**
     * @brief Final recostructed scene with camera extrinsics and reconstructed 3d points
     */
    struct SfMResult
    {
        std::vector<Mat4> extrinsics; // Camera extrinsics
        std::vector<Vec3> points;     // Reconstructed 3D points
    };

    struct EightPointResult
    {
        Mat4 pose;                // Pose of camera 2
        std::vector<Vec3> points; // Reconstructed 3d points
    };
} // Namespace SfM