#pragma once
#include <Eigen/Dense>
#include <vector>
#include <memory>

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

    using Vec2I = Eigen::Matrix<int, 2, 1>;
    using Vec3rgb = Eigen::Matrix<unsigned char, 3, 1>;

    constexpr REAL EPSILON = static_cast<REAL>(1e-6);

    /**
     * @brief Wrapper for an image.
     * @param data Holds the image data in a T*. The data is stored in the following way [T1, T2, T3, ...] from top left to bottom right row by row.
     * If T is uchar then it is [r1, g1, b1, r2, g2, b2, ...]
     * @param width Width of the image
     * @param height Height of the image
     */
    template <typename T>
    class Image
    {
    public:
        Image(int w, int h) : width{w}, height{h}
        {
            data = new T[w * h];
        }

        /**
         * @brief Use this if you have multiple channels per pixel, e.g. rgb
         */
        Image(int w, int h, int dim) : width{w}, height{h}
        {
            data = new T[w * h * dim];
        }

        ~Image()
        {
            delete[] data;
        }

        inline T &at(int x, int y)
        {
            return data[y * width + x];
        }

        // Move semantics for efficient returns.
        Image(Image &&other) noexcept
            : data(other.data), width(other.width), height(other.height)
        {
            other.data = nullptr;
            other.width = 0;
            other.height = 0;
        }

        Image &operator=(Image &&other) noexcept
        {
            if (this != &other)
            {
                delete[] data; // Free current data

                data = other.data;
                width = other.width;
                height = other.height;

                other.data = nullptr;
                other.width = 0;
                other.height = 0;
            }
            return *this;
        }

        Image(const Image &) = delete;
        Image &operator=(const Image &) = delete;

        T *data; // Use T* instead of std::vector<T> because I want to malloc without initializing the data which that takes a long time
        int width;
        int height;
    };

    /**
     * @brief An observation represents one 2D measurement
     * @param point 2D measurement of the point in pixel coordinates
     * @param trackId Unique Id per Track
     * @param indexInLastFrame Matching Observation in the previous Frame
     * @param inlier Bool is false if RANSAC marks this point as an outlier
     * @param wasOutlierBefore Helper bool to check if the obervation is really an outlier or just a new track 
     */
    struct Observation
    {
        enum Status
        {
            UNINITIALIZED = -1,
            NOT_FOUND = -2
        };

        Vec2 point;
        int trackId;
        int indexInLastFrame = UNINITIALIZED;
        bool inlier = true;
        bool wasOutlierBefore = false;
    };

    /**
     * @brief A keypoint represents a point feature found by a keypoint detector such as SIFT, SURF, ...
     * @param descriptor Feature descriptor (For SIFT descriptor.size() = 128)
     * @param point Sub-pixel coordinates of the feature
     * @param size Feature diameter, meaningful neighborhood (For SIFT: zoom-level where the feature was found)
     * @param angle Feature orientation [rad]
     * @param response Keypoint detector response of the feature, strength of the feature
     * @param octave Pyramid octave in which the keypoint has been detected
     * @param trackId Unique Id per Track
     * @param observation Pointer to observation in Frames
     */
    struct Keypoint
    {
        enum Status
        {
            UNINITIALIZED = -1,
        };

        std::vector<float> descriptor;
        Vec2 point;
        REAL size;
        REAL angle;
        REAL response;
        int octave;
        int trackId = UNINITIALIZED;
        Observation *observation = nullptr;
    };

    /**
     * @brief An approximation represents one 3d point (over multiple frames)
     * @param point 3d approximation of the point
     * @param color Color of the track
     */
    struct Approximation
    {
        Vec3 point;
        Vec3rgb color;
    };

    /**
     * @brief A frame from the camera with an unique id and a bunch of observations. For the "horizontal" approach, one frame one data object
     * @param observation Vector of Observations (Sorted by trackId)
     * @param frameId Unique Id per Frame
     */
    struct Frame
    {
        std::vector<std::unique_ptr<Observation>> observations;
        int frameId;
    };

    /**
     * @brief Represents the result of the eight point algorithm
     * @param pose 4x4 view Matrix of camera 2
     * @param points Vector of reconstructed points
     */
    struct EightPointResult
    {
        Mat4 pose;
        std::vector<Vec3> points;
    };

    /**
     * @brief Final recostructed scene with camera extrinsics and reconstructed 3d points
     * @param extrinsics Vector of camera extrinsics
     * @param points Vector of reconstructed points (Point with trackId i is stored in points[i])
     * @param inlierMask Vector of bools representing if a point is an inlier or an outlier
     */
    struct SfMResult
    {
        std::vector<Mat4> extrinsics;
        std::vector<Vec3> points;
        std::vector<bool> inlierMask;
    };

    // ------------------------------------------------------------------------------------------------------------
    // Not used from here -----------------------------------------------------------------------------------------
    // ------------------------------------------------------------------------------------------------------------

    /**
     * @brief An observation represents one 2D measurement
     */
    struct SimpleObservation
    {
        int frameId; // Frame id
        Vec2 point;  // 2D measurement in that frame
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
} // Namespace SfM