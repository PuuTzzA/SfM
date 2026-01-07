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

    constexpr REAL EPSILON = static_cast<REAL>(1e-7);

    /**
     * @brief An observation represents one 2D measurement
     */
    struct Observation
    {
        enum Status
        {
            UNINITIALIZED = -1,
            NOT_FOUND = -2
        };

        Vec2 point;                           // 2D measurement of the point in pixel coordinates
        int trackId;                          // Unique Id per Frame
        int indexInLastFrame = UNINITIALIZED; // Matching Observation in the previous Frame
    };

    /**
     * @brief A frame from the camera with an unique id and a bunch of observations. For the "horizontal" approach, one frame one data object
     */
    struct Frame
    {
        std::vector<Observation> observations; // Vector of Observations
        int frameId;                           // Unique Id per Frame
    };

    /**
     * @brief An observation represents one 2D measurement
     */
    struct SimpleObservation
    {
        int frameId; // Frame id
        Vec2 point;  // 2D measurement in that frame
    };

    /**
     * @brief An approximation represents one 3d point (over multiple frames)
     */
    struct Approximation
    {
        int trackId; // Track id
        Vec3 point;  // 3d approximation of that point
    };

    /**
     * @brief A Track a track represents one physical 3D point. For the "vertical" approach, one 3d point one data object, but one frame = many 3d points = many data objects
     *
     * @param id Unique id for the 3d point
     * @param observations Vector of Observation, all frames where the 3d point is seen (Sorted by frameId)
     *
     * e.g.:
     * Track #42:
     *  frame 0 → (120.3, 85.1)
     *  frame 1 → (118.9, 84.7)
     *  frame 3 → (115.2, 83.9)
     */
    struct Track
    {
        int id;                                      // stable identity
        int lastIndex;                               // helper variable to find matches faster
        std::vector<SimpleObservation> observations; // sorted by frame Id
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