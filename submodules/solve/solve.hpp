#pragma once
#include "../SfM.hpp"

namespace SfM::solve
{
    /**
     * @brief Solves for the camera extrinsics and 3d positions of the keypoints with the 8 point algorithm.
     * 
     * @param tracks Vector of tracked keypoints
     * @param K Camera intrinsics
     * 
     * @return Camera extrinsics and 3d locations of the keypoints
     */
    SfMResult eightPointAlgorithm(const std::vector<Track> &tracks, Mat3 K);

    EightPointResult eightPointAlgorithm(const std::vector<Vec2> &points1, const std::vector<Vec2> &points2);
} // Namespace SfM::solve