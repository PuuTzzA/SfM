#include <iostream>
#include "submodules/calibrate/calibrate.hpp"
#include "submodules/detect/detect.hpp"
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
#include <numeric>

int main()
{
    // Testing
    auto images = SfM::io::loadImages("../calibration_data/test_dir");
    std::cout << "image count: " << images.size() << std::endl;

    auto calibration = SfM::calibrate::calibrateCamera(images);

    cv::imshow("Distorted", images[0]);
    cv::imshow("Undistorted", SfM::calibrate::undistort(images[0], calibration));
    cv::waitKey(0);


    // std::string path = "../../Data/real_image.jpg";
    std::string path = "../../Data/calibration.jpg";

    SfM::Image<uchar> immg = SfM::io::loadImage(path);
    SfM::Image<float> immggray = SfM::util::rgbToREAL<float>(immg);
    auto s = std::chrono::steady_clock::now();

    auto blurred = SfM::util::gaussianBlur(immggray, SfM::Vec2(5, 5));

    auto d = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - s).count();

    std::cout << "Took " << d << "ms" << std::endl;

    blurred = SfM::util::mulScalar(blurred, static_cast<float>(255));

    auto cv = SfM::io::imageToCvMat(blurred);
    cv::imwrite("../../Data/______blurred.png", cv);

    return 0;
    /* auto startLoadCv = std::chrono::steady_clock::now();
    std::cout << "Loading image with OpenCv" << std::endl;
    cv::Mat img = cv::imread(path);
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB); // Convert BGR to RGB
    auto vecimg = SfM::io::cvMatToVector(img);

    std::cout << "Took " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - startLoadCv).count() << "ms" << std::endl;

    auto startLoad = std::chrono::steady_clock::now();

    auto vecimg2 = SfM::io::loadImage(path);

    std::cout << "Took " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - startLoad).count() << "ms" << std::endl;

    bool areSimilar = true;
    for (size_t i = 0; i < vecimg.size(); i++)
    {
        if (std::abs(vecimg[i] - vecimg2.data[i]) > 2)
        {
            areSimilar = false;
            std::cout << "difference at " << i << " of " << std::abs(vecimg[i] - vecimg2.data[i]) << std::endl;
            break;
        }
    } */

    cv::Mat img = cv::imread(path);

    auto cornersCV = SfM::detect::harrisCornerDetectionSubPixelOpenCv(img);
    std::vector<int> millis0;

    for (int _ = 0; _ < 20; _++)
    {
        auto startDetectCV = std::chrono::steady_clock::now();
        SfM::detect::harrisCornerDetectionOpenCv(img);
        auto dur = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - startDetectCV).count();
        millis0.push_back(dur);
        std::cout << _ << ": Took " << dur << "ms" << std::endl;
    }

    std::sort(cornersCV.begin(), cornersCV.end(),
              [](const SfM::Vec2 &u, const SfM::Vec2 &v)
              {
                  return u[0] < v[0];
              });

    SfM::Image<uchar> image2 = SfM::io::loadImage(path);

    std::vector<int> millis1;
    for (int _ = 0; _ < 20; _++)
    {
        auto startDetect2 = std::chrono::steady_clock::now();
        auto corners2 = SfM::detect::harrisCornerDetection(image2, 2);
        auto dur2 = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - startDetect2).count();
        millis1.push_back(dur2);
        std::cout << _ << ": Took " << dur2 << "ms" << std::endl;
    }
    std::cout << "OpenCV: average" << std::reduce(millis0.begin(), millis0.end()) / millis0.size() << std::endl;
    std::cout << "MyNew:  average" << std::reduce(millis1.begin(), millis1.end()) / millis1.size() << std::endl;

    /* std::sort(corners.begin(), corners.end(),
              [](const SfM::Vec2 &u, const SfM::Vec2 &v)
              {
                  return u[0] < v[0];
              });

    // std::cout << "len of cornersCv: " << cornersCV.size() << ", len of corners: " << corners.size() << std::endl;

     for (int i = 0; i < cornersCV.size(); i++)
     {
         auto c = cornersCV[i];
         // std::cout << "cv: corner at: " << c[0] << ", " << 1080 - c[1] << std::endl;
     }

    // SfM::detect::harrisCornerDetectionOpenCv(img);

    // std::cout << "OpenCV version: " << CV_VERSION << std::endl;
    return 0;

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

    auto frames = SfM::test::generateRandomPointsFrames(cameraExtrinsics, K, SfM::Vec3(0, 7, 0), SfM::Vec3(2, 2, 1), points, numPoints, 0.9, SfM::Vec2(1, 1));
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

    // bundle adjustment
    /* auto start2 = std::chrono::high_resolution_clock::now();

    auto resultBundle = SfM::solve::bundleAdjustment(frames, K, numPoints, nullptr, SfM::util::cvCameraToBlender(SfM::util::calculateTransformationMatrixDeg(90, 0, 0, SfM::Vec3(0, 0, 0))));

    auto end2 = std::chrono::high_resolution_clock::now();
    auto duration2 = std::chrono::duration_cast<std::chrono::milliseconds>(end2 - start2);
    std::cout << "Execution time: " << duration2.count() << " ms" << std::endl;

    SfM::io::exportTracksForBlender(resultBundle.extrinsics, resultBundle.points, "../../Data/test0_bundle.txt"); */

    return 0;
}
