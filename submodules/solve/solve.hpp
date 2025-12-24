#pragma once
#include "../SfM.hpp"

namespace SfM::solve
{
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
     * @brief Calculates the view matrix and 3d positions between two frames
     */
    EightPointResult eightPointAlgorithm(const std::vector<Vec2> &points1, const std::vector<Vec2> &points2);
} // Namespace SfM::solve