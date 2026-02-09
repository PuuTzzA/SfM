#include <iostream>
#include "submodules/SfM.hpp"
#include "submodules/scene.hpp"
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

// #define CALIBRATION

int main()
{
#ifdef CALIBRATION
    std::string pathToImages = "../../Data/S21/calibration_small/";
    auto images = SfM::io::loadImages(pathToImages);
    std::cout << "image count: " << images.size() << std::endl;

    SfM::io::storeCalibration("../../Data/S21/calibration.json", SfM::calibrate::calibrateCamera(images, {6, 8}));

    auto calibration = SfM::io::loadCalibration("../../Data/S21/calibration.json");

    std::cout << "Camera Matrix:\n"
              << calibration.matrix << std::endl;
    std::cout << "Distortion Coefficients:\n"
              << calibration.distortionCoeffs << std::endl;
#else
    // Explicit definition of the intrinsics Matrix
    /* SfM::REAL width_px = 1920.;
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
        0, 0, 1; */

    auto calibration = SfM::io::loadCalibration("../../Data/S21/calibration.json");
    // auto calibration = SfM::io::loadCalibration("../calibration_data/fr1_calib.json");

    SfM::Mat3 K = calibration.K;

    std::cout << K << std::endl;

    SfM::Mat4 startTransform = SfM::util::cvCameraToBlender(SfM::util::calculateTransformationMatrixDeg(65, 0, 0, SfM::Vec3(0, 0, 0)));
    // SfM::Mat4 startTransform = SfM::Mat4::Identity();

    SfM::match::MATCHING_OPTIONS matchingOptions{
        .threshold = 0.95,
        .maxDistancePxSquared = 100 * 100,
    };
    SfM::solve::RANSAC_OPTIONS ransacOptions{
        .maxIter = 2048 * 4,
        .maxTimeMs = 20000,
        .maxSquaredError = 1,
        .successProb = 0.999,
    };
    SfM::solve::BUNDLE_ADJUSTMENT_OPTIONS baOptions{
        .ceresOptions = {
            .trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT, // LEVENBERG_MARQUARDT (better?) is the default, other is DOGLEG
            .max_num_iterations = 512,
            .max_solver_time_in_seconds = 100,
            .num_threads = static_cast<int>(std::thread::hardware_concurrency()),
            .max_num_consecutive_invalid_steps = 10,
            // .linear_solver_type = ceres::DENSE_SCHUR, // (DENSE_SCHUR and SPARSE_SCHUR best for BA) http://ceres-solver.org/nnls_solving.html#linear-solvers
            .linear_solver_type = ceres::SPARSE_SCHUR,
            .minimizer_progress_to_stdout = true,
        },
        .printSummary = true,
        .useLiftingScheme = false,
    };
    SfM::SCENE_OPTIONS sceneOptions{
        .matchingOptions = matchingOptions,
        .ransacOptions = ransacOptions,
        .bundleAdjustmentOptions = baOptions,
        .useEightPoint = true,
        .splitTracks = false,
        .useRANSAC = true,
        .verbose = true,
    };

    SfM::Scene scene(K, startTransform, sceneOptions);

    // std::string pathToImages = "../../Data/rgbd_dataset_freiburg1_xyz/rgb";
    std::string pathToImages = "../../Data/S21/krabbe_small";

    auto images = SfM::io::loadImages(pathToImages, 0, 27);

    // Don't do this for phone images, since they are already corrected and applying this would yield incorrect results
    /* for (auto &img : images)
    {
        img = SfM::calibrate::undistort(img, calibration);
    } */

    std::cout << "loaded " << images.size() << " images." << std::endl;
    for (int i = 0; i < images.size(); i++)
    {
        auto keypoints = SfM::detect::SIFTOpenCv(images[i]);
        std::cout << "Detected: " << keypoints.size() << " keypoints." << std::endl;

        scene.pushBackImageWithKeypoints(std::move(images[i]), std::move(keypoints));
    }

    scene.optimizeExtrinsicsAnd3dPoints();

    SfM::io::storeImages(scene, "../../Data/S21/krabbe", "krabbe_");
    SfM::io::exportSceneForBlender(scene, "../../Data/S21/krabbe.json", "./krabbe");
    // SfM::io::storeImages(scene, "../../Data/fr1/images", "fr1_");
    // SfM::io::exportSceneForBlender(scene, "../../Data/fr1/fr1.json", "./images");
    // SfM::io::exportTrack(scene.getExtrinsics(), timestamps, "../../Data/fr1_track.txt", "../../Data/rgbd_dataset_freiburg1_xyz/groundtruth.txt");
#endif
}
