#include <iostream>
#include "submodules/io/file.hpp"
#include "submodules/solve/solve.hpp"
#include "submodules/io/blender.hpp"
#include "submodules/test/generate.hpp"
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <optional>

int main()
{
    std::vector<SfM::Mat4> cameraExtrinsics{
        SfM::test::calculateMatrix(90, 0, 0, SfM::Vec3(0, 0, 0)),
        SfM::test::calculateMatrix(87, 0, 5, SfM::Vec3(0.5, 0, 0.2)),
        SfM::test::calculateMatrix(83, 0, 10, SfM::Vec3(1.0, 0, 0.8)),
        SfM::test::calculateMatrix(80, 0, 15, SfM::Vec3(1.5, 0, 1)),
        SfM::test::calculateMatrix(69, 7, 28, SfM::Vec3(4.02, 0.405, 2.82))};

    SfM::REAL width_px = 1920.;
    SfM::REAL height_px = 1080.;
    SfM::REAL f_mm = 50.;
    SfM::REAL sensor_mm = 36.f;

    SfM::REAL fx = f_mm * width_px / sensor_mm; // focal length in pixel
    SfM::REAL fy = fx;
    SfM::REAL cx = width_px / 2.0f;  // 960
    SfM::REAL cy = height_px / 2.0f; // 540

    SfM::Mat3 K;
    K << fx, 0, cx,
        0, fy, cy,
        0, 0, 1;

    std::vector<SfM::Vec3> points{};

    auto tracks = SfM::test::generateRandomPoints(cameraExtrinsics, K, SfM::Vec3(0, 7, 0), SfM::Vec3(2, 2, 1), points);

    { // Export Ground Truth values
        // Convert from Blender space to Computer Vision space bc SfM::io::exportTracksForBlender expects io
        for (auto &extrinsic : cameraExtrinsics)
        {
            SfM::Mat4 blenderToCv = SfM::Mat4::Identity();
            blenderToCv(1, 1) = -1;
            blenderToCv(2, 2) = -1;

            extrinsic *= blenderToCv;
        }

        SfM::io::exportTracksForBlender(cameraExtrinsics, points, "../../Data/test0.txt");
    }

    /* for (auto &t : tracks)
    {
        std::cout << t.observations[0].point << std::endl;
    } */

    // auto tracks = SfM::io::loadTrackedPoints("../../Data/Suzanne/tracks.txt");

    // SfM::io::printForBlender(SfM::solve::eightPointAlgorithm(tracks));

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
