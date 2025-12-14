#include <iostream>
#include "submodules/file.h"
#include "submodules/solve.h"
#include "submodules/blender.h"
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

int main()
{
    std::cout << "Hello SfM!" << std::endl;

    auto tracks = SfM::File::loadTrackedPoints("../../Data/Suzanne/tracks.txt");

    SfM::Blender::printForBlender(SfM::Solve::eightPointAlgorithm(tracks));

    // SfM::File::drawPointsOnImage("../../Data/Suzanne/susanne_0001.png", "ouput.png", tracks[0]);

    // std::vector<std::string> paths({"../../Data/Suzanne/susanne_0001.png", "../../Data/Suzanne/susanne_0002.png", "../../Data/Suzanne/susanne_0003.png", "../../Data/Suzanne/susanne_0004.png"});
    // SfM::File::drawCollageWithTracks(paths, tracks, 0, 1, "collage.png");

    /* for (auto &point : tracks)
    {
        std::cout << "pointlolol" << std::endl;

        for (auto &frame : point)
        {
            std::cout << frame[0] << ", " << frame[1] << std::endl;
        }
    } */

    /* cv::Mat img = cv::imread("../../Data/Suzanne/susanne_0001.png");

    if (img.empty())
    {
        std::cout << "Could not read the image: " << std::endl;
        return 1;
    }

    cv::imshow("Display window", img);
    int k = cv::waitKey(0); // Wait for a keystroke in the window */

    std::cout << "OpenCV version: " << CV_VERSION << std::endl;
    return 0;
}
