#include <iostream>
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
    int W = 1920 * 2;
    int H = 1080 * 2;

    SfM::Image<float> img1;
    img1.width = W;
    img1.height = H;
    img1.data.resize(W * H);

    SfM::Image<float> img2;
    img2.width = W;
    img2.height = H;
    img2.data.resize(W * H);

    for (int i = 0; i < W * H; i++)
    {
        img1.data[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        img2.data[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }

    std::vector<int> millis;
    std::vector<int> millis2;
    std::vector<int> millis3;
    std::vector<int> millis4;
    std::vector<int> millis5;
    for (int _ = 0; _ < 100; _++)
    {
        auto startDetect = std::chrono::steady_clock::now();

        auto img3 = SfM::util::sub<float>(img1, img2);

        auto dur = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - startDetect).count();
        millis.push_back(dur);
        std::cout << _ << ".1: Took " << dur << "ms" << std::endl;
        delete (img3);

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        auto startDetect2 = std::chrono::steady_clock::now();

        auto img4 = SfM::util::sub2(img1, img2);

        auto dur2 = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - startDetect2).count();
        millis2.push_back(dur2);
        std::cout << _ << ".2: Took " << dur2 << "ms" << std::endl;
        delete (img4);

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        auto startDetect3 = std::chrono::steady_clock::now();

        auto img5 = SfM::util::sub3(img1, img2);

        auto dur3 = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - startDetect3).count();
        millis3.push_back(dur3);
        std::cout << _ << ".3: Took " << dur3 << "ms" << std::endl;
        delete (img5);

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        auto startDetect4 = std::chrono::steady_clock::now();

        auto img6 = SfM::util::sub4(img1, img2);

        auto dur4 = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - startDetect4).count();
        millis4.push_back(dur4);
        std::cout << _ << ".4: Took " << dur4 << "ms" << std::endl;
        delete(img6);

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        auto startDetect5 = std::chrono::steady_clock::now();

        auto img7 = SfM::util::sub5(img1, img2);

        auto dur5 = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - startDetect5).count();
        millis5.push_back(dur5);
        std::cout << _ << ".5: Took " << dur5 << "ms" << std::endl;
        delete(img7);
    }

    std::cout << "OpenMP1: average" << std::reduce(millis.begin(), millis.end()) / millis.size() << std::endl;
    std::cout << "OpenMP2: average" << std::reduce(millis2.begin(), millis2.end()) / millis2.size() << std::endl;
    std::cout << "TBB 1  : average" << std::reduce(millis3.begin(), millis3.end()) / millis3.size() << std::endl;
    std::cout << "TBB 2  : average" << std::reduce(millis4.begin(), millis4.end()) / millis4.size() << std::endl;
    std::cout << "Both   : average" << std::reduce(millis5.begin(), millis5.end()) / millis5.size() << std::endl;

    return 0;
    std::string path = "../../Data/real_image.jpg";

    /*
        auto startLoadCv = std::chrono::steady_clock::now();
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

    /* cv::Mat img = cv::imread(path);

    auto cornersCV = SfM::detect::harrisCornerDetectionSubPixelOpenCv(img);
    auto startDetectCV = std::chrono::steady_clock::now();
    SfM::detect::siftCv(img);
    std::cout << "Took " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - startDetectCV).count() << "ms" << std::endl;

    std::sort(cornersCV.begin(), cornersCV.end(),
              [](const SfM::Vec2 &u, const SfM::Vec2 &v)
              {
                  return u[0] < v[0];
              });

    SfM::Image image = SfM::io::loadImage(path);

    std::vector<int> millis;
    for (int _ = 0; _ < 1; _++)
    {
        auto startDetect = std::chrono::steady_clock::now();
        auto corners = SfM::detect::harrisCornerDetection(image, 2);
        auto dur = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - startDetect).count();
        millis.push_back(dur);
        std::cout << _ << ": Took " << dur << "ms" << std::endl;
    }

    std::cout << "average" << std::reduce(millis.begin(), millis.end()) / millis.size() << std::endl; */

    /* std::sort(corners.begin(), corners.end(),
              [](const SfM::Vec2 &u, const SfM::Vec2 &v)
              {
                  return u[0] < v[0];
              }); */

    // std::cout << "len of cornersCv: " << cornersCV.size() << ", len of corners: " << corners.size() << std::endl;

    /*  for (int i = 0; i < cornersCV.size(); i++)
     {
         auto c = cornersCV[i];
         // std::cout << "cv: corner at: " << c[0] << ", " << 1080 - c[1] << std::endl;
     }
  */
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
