#pragma once
#include "../SfM.hpp"

namespace SfM::solve
{
    /**
     * @brief Solves for the camera extrinsics and 3d positions of the keypoints with the 8 point algorithm.
     *
     * @param frames Vector of frames with keypoints
     * @param K Camera intrinsics
     * @param numTotKeypoints Total Number of unique tracks (3d points)
     *
     * @return Camera extrinsics and 3d locations of the keypoints
     */
    SfMResult eightPointAlgorithm(std::vector<Frame> &frames, Mat3 K, const int numTotKeypoints);

    /**
     * @brief Solves for the camera extrinsics and 3d positions of the keypoints with the 8 point algorithm.
     *
     * @param tracks Vector of tracked keypoints
     * @param K Camera intrinsics
     * @param numFrames Number of images in the sequence
     *
     * @return Camera extrinsics and 3d locations of the keypoints
     */
    SfMResult eightPointAlgorithm(std::vector<Track> &tracks, Mat3 K, const int numFrames);

    /**
     * @brief Solves for the camera extrinsics and 3d positions of the keypoints using non-linear optimization (bundle adjustment)
     * 
     * @param frames Vecotr of frames with keypoints
     * @param K Camera intrinsics
     * @param numTotKeypoints Total Number of uniquze tracks (3d points)
     * 
     * @return Camera extrinsics and 3d locations of the keypoints
     */
    SfMResult bundleAdjustment(std::vector<Frame> &frames, Mat3 K, const int numTotKeypoints);
    
    /**
     * @brief Calculates the view matrix and 3d positions between two frames
     */
    EightPointResult eightPointAlgorithm(const std::vector<Vec2> &points1, const std::vector<Vec2> &points2);
} // Namespace SfM::solve