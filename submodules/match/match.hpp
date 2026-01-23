#pragma once
#include "../SfM.hpp"
#include <opencv2/opencv.hpp>

namespace SfM::match
{
    /**
     * @brief Creates a new frame from a list of Keypoints
     * @param keypoints Vector of Keypoints
     * @return A newly initialized frame
     */
    Frame createFirstFrameFromKeypoints(std::vector<Keypoint> &keypoints);

    /**
     * @brief Creates a new frame by matching keypoints to a existing frame
     * The matching is performed on a two-sided
     * @param frame0 Previous already matched frame
     * @param keypoints1 New vector of keypoints
     * @return Tuple of <Frame matched to the previous one, amount of not matched keypoints i.e. new trackIds>
     */
    std::tuple<Frame, int> matchTwoSided(Frame &frame0, std::vector<Keypoint> &keypoints1);

    // vl hungarian method

} // Namespace SfM::match