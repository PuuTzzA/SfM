#include <iostream>
#include "submodules/io/file.hpp"
#include "submodules/solve/solve.hpp"
#include "submodules/io/blender.hpp"
#include "submodules/test/generate.hpp"
#include "submodules/util/util.hpp"
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <optional>
#include <chrono>

int main()
{
    std::vector<SfM::Mat4> cameraExtrinsics{
        SfM::util::calculateTransformationMatrixDeg(90, 0, 0, SfM::Vec3(0, 0, 0)),
        SfM::util::calculateTransformationMatrixDeg(87, 0, 5, SfM::Vec3(0.5, 0, 0.2)),
        SfM::util::calculateTransformationMatrixDeg(83, 0, 10, SfM::Vec3(1.0, 0, 0.8)),
        SfM::util::calculateTransformationMatrixDeg(80, 0, 15, SfM::Vec3(1.5, 0, 1)),
        SfM::util::calculateTransformationMatrixDeg(69, 7, 28, SfM::Vec3(4.02, 0.405, 2.82))};

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
    int numPoints = 50;

    auto frames = SfM::test::generateRandomPointsFrames(cameraExtrinsics, K, SfM::Vec3(0, 7, 0), SfM::Vec3(2, 2, 1), points, numPoints, 0.9, SfM::Vec2(2, 2));
    numPoints += SfM::test::addOutliersToFrames(frames, 1, 10, numPoints);

    // Export Ground Truth values
    for (auto &e : cameraExtrinsics)
    {
        e = SfM::util::cvCameraToBlender(e);
    }
    for (auto &p : points)
    {
        p = SfM::util::blendCvMat3() * p;
    }
    SfM::io::exportTracksForBlender(cameraExtrinsics, points, "../../Data/test0.txt");

    // 8 point algorithm
    auto start = std::chrono::high_resolution_clock::now();

    auto resultEight = SfM::solve::eightPointAlgorithm(frames, K, numPoints, SfM::util::cvCameraToBlender(SfM::util::calculateTransformationMatrixDeg(90, 0, 0, SfM::Vec3(0, 0, 0))));

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Execution time: " << duration.count() << " ms" << std::endl;

    SfM::io::exportTracksForBlender(resultEight.extrinsics, resultEight.points, "../../Data/test0_8point.txt");

    /* // bundle adjustment
    start = std::chrono::high_resolution_clock::now();

    auto resultBundle = SfM::solve::bundleAdjustment(frames, K, numPoints, nullptr, SfM::util::cvCameraToBlender(SfM::util::calculateTransformationMatrixDeg(90, 0, 0, SfM::Vec3(0, 0, 0))));

    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Execution time: " << duration.count() << " ms" << std::endl;

    SfM::io::exportTracksForBlender(resultBundle.extrinsics, resultBundle.points, "../../Data/test0_bundle.txt"); */

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
