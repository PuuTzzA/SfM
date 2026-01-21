#pragma once

#include "../SfM"

#include <vector>
#include <opencv2/opencv.hpp>

namespace SfM::calibrate {

    struct CameraCalibration {
        Vec3 radialDistortion;
        Vec2 tangentialDistortion;

        Vec2 focalLength;
        Vec2 opticalCenters;

        /**
         * @returns Mat3 representing the intrinsic matrix of the camera.
         */
        inline Mat3 getMatrix() {
            Mat3 m;
            m << focalLength[0], 0, opticalCenters[0], 0, focalLength[1], opticalCenters[1], 0, 0, 1;
            return m;
        }
    }
} // namespace Sfm::calibrate