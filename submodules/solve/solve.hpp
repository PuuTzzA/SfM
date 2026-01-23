#pragma once
#include "../SfM.hpp"
#include <functional>

namespace SfM::solve
{
    /**
     * @brief Solves for the camera extrinsics and 3d positions of the keypoints with the 8 point algorithm.
     *
     * @param frames Vector of frames with obersations
     * @param K Camera intrinsics
     * @param numTotTracks Total Number of unique tracks (3d points)
     * @param startTransform (default Mat4::Identity()) Global transform applied to everything
     *
     * @return Camera extrinsics and 3d locations of the 3d points
     */
    SfMResult eightPointAlgorithm(std::vector<Frame> &frames, const Mat3 K, const int numTotTracks, const Mat4 startTransform = Mat4::Identity());

    /* /**
     * REMOVED, because I din't want to maintain two separate Instances of the same logic
     * @brief Solves for the camera extrinsics and 3d positions of the keypoints with the 8 point algorithm.
     *
     * @param tracks Vector of tracked tracks
     * @param K Camera intrinsics
     * @param numFrames Number of images in the sequence
     * @param startTransform (default Mat4::Identity()) Global transform applied to everything
     *
     * @return Camera extrinsics and 3d locations of the 3d points
     */
    // SfMResult eightPointAlgorithm(std::vector<Track> &tracks, const Mat3 K, const int numFrames, const Mat4 startTransform = Mat4::Identity()); */

    /**
     * @brief Solves for the camera extrinsics and 3d positions of the keypoints using non-linear optimization (bundle adjustment).
     * @note Very very sensitive to the starting initialization!
     *
     * @param frames Vector of frames with observations
     * @param K Camera intrinsics
     * @param numTotTracks Total number of uniquze tracks (3d points)
     * @param initialGuess Initial guess for BA
     * @param startTransform (default Mat4::Identity()) Global transform applied to everything
     *
     * @return Camera extrinsics and 3d locations of the 3d points
     */
    SfMResult bundleAdjustment(const std::vector<Frame> &frames, const Mat3 K, const int numTotTracks, const SfMResult *initialGuess, const Mat4 startTransform = Mat4::Identity());

    /**
     * @brief Calculates the view matrix and 3d positions between two frames
     */
    EightPointResult eightPointAlgorithm(const std::vector<Vec2> &points1, const std::vector<Vec2> &points2);

    /**
     * @brief Calculates the view matrix from calcPoints1 and calcPoints2, but estimates the 3d location of allPoint
     */
    EightPointResult eightPointAlgorithmFromSubset(const std::vector<Vec2> &calcPoints1, const std::vector<Vec2> &calcPoints2,
                                                   const std::vector<Vec2> &allPoints1, const std::vector<Vec2> &allPoints2);

    /**
     * @brief Options for random sample consensus to find in- and outliers
     *
     * @param minN Minimum number of data points to estimate parameters
     * @param maxIter Maximum iteraions (permutations tried)
     * @param maxTimeMs [ms] Maximum time spend trying to find the best estimate
     * @param maxSquaredError [px] Threshold value to determine if points are fit well
     * @param successProb Probability of success
     * @param model Function that fits the data
     * @param loss Function to calculate the loss of a fit data point
     */
    struct RANSAC_OPTIONS
    {
        using fittingFunction = std::function<EightPointResult(const std::vector<Vec2> &, const std::vector<Vec2> &, const std::vector<Vec2> &, const std::vector<Vec2> &)>; // calcPoints1, calcPoints2, allPoints1, allPoints2
        using lossFunction = std::function<REAL(const Mat3 &, const Vec2, const Vec2, const Vec3, const Mat4 &)>;                                                            // Intrinsics, Observation1, Observation2, 3dPoint, View Matrix

        int minN;
        int maxIter;
        int maxTimeMs;
        REAL maxSquaredError;
        REAL successProb = static_cast<REAL>(0.99);
        fittingFunction model;
        lossFunction loss;
    };

    /**
     * @brief Finds largest subset of inliers with RANSAC
     *
     * @param x Vector of observations
     * @param y Vector of observations in next frame
     * @param K Intrinsics matrix
     * @param options Options for the algorithm
     *
     * @return Vector of inliers
     */
    std::vector<int> RANSAC(const std::vector<Vec2> &x, const std::vector<Vec2> &y, const Mat3 &K, const RANSAC_OPTIONS &options);

    /**
     * @brief Calculates the reprojection error of a 3d point given the NORMALIZED observation, viewMatrix and the intrinsics
     * @return Squared distance
     */
    inline REAL reprojectionError(const Mat3 &K, const Vec2 observation, const Vec3 point3d, const Mat4 &viewMatrix)
    {
        Vec3 p = K * (viewMatrix * point3d.homogeneous()).head<3>();
        p[0] /= p[2];
        p[1] /= p[2];
        Vec3 obs(observation[0], observation[1], 1.);
        obs = K * obs; // multiply by K because the observations are normalized, don't divie by z, bc z == 1.
        REAL dx = p[0] - obs[0];
        REAL dy = p[1] - obs[1];
        return dx * dx + dy * dy;
    }

    /**
     * @brief Computes the thing that gets minimized in the 8 point algorithm. Namely: x2^T * E * x1 = 0
     */
    inline REAL eightPointError(const Mat4 &viewMat, const Vec2 observation1, const Vec2 observation2)
    {
        Mat3 R = viewMat.block<3, 3>(0, 0);
        Vec3 t = viewMat.block<3, 1>(0, 3);

        Mat3 T;
        T << 0, -t[2], t[1],
            t[2], 0, -t[0],
            -t[1], t[0], 0;

        Vec3 x2(observation2[0], observation2[1], 1.);
        Vec3 x1(observation1[0], observation1[1], 1.);
        // x2 = m_K * x2; This does not work ... therefore I don't know how to make this usefull, since no units
        // x1 = m_K * x1;
        return x2.transpose() * (T * R) * x1;
    }
} // Namespace SfM::solve