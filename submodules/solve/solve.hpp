#pragma once
#include "../SfM.hpp"

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

    /**
     * @brief Solves for the camera extrinsics and 3d positions of the keypoints with the 8 point algorithm.
     *
     * @param tracks Vector of tracked tracks
     * @param K Camera intrinsics
     * @param numFrames Number of images in the sequence
     * @param startTransform (default Mat4::Identity()) Global transform applied to everything
     *
     * @return Camera extrinsics and 3d locations of the 3d points
     */
    SfMResult eightPointAlgorithm(std::vector<Track> &tracks, const Mat3 K, const int numFrames, const Mat4 startTransform = Mat4::Identity());

    /**
     * @brief Solves for the camera extrinsics and 3d positions of the keypoints using non-linear optimization (bundle adjustment)
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

    /* REAL reprojectionError(const Mat4 &K, const Vec2 observation, const Vec3 point3d, const Mat4 &extrinsics)
    {
        Vec3 p = (K * extrinsics.inverse() * point3d.homogeneous()).head<3>();
        p[0] /= p[2];
        p[1] /= p[2];
        REAL dx = p[0] - observation[0];
        REAL dy = p[1] - observation[1];
        return dx * dx + dy * dy;
    } */
} // Namespace SfM::solve