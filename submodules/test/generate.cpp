#include "generate.hpp"
#include <iostream>
#include <math.h>

namespace SfM::test
{
    std::vector<Track> generateRandomPoints(std::vector<Mat4> &cameraExtrinsics,
                                            Mat3 cameraIntrinsics,
                                            Vec3 pointsLocation,
                                            Vec3 pointsRadius,
                                            std::optional<std::reference_wrapper<std::vector<Vec3>>> points,
                                            int numPoints,
                                            Vec2 detectionError)
    {
        std::vector<Track> tracks;
        tracks.reserve(numPoints);

        std::vector<Vec3> ps{};
        std::vector<Vec3> &points3d = points ? points->get() : ps;
        points3d.reserve(numPoints);

        for (int i = 0; i < numPoints; i++)
        {
            REAL rx = static_cast<REAL>(2) * pointsRadius[0] * static_cast<REAL>(rand()) / static_cast<REAL>(RAND_MAX) - pointsRadius[0];
            REAL ry = static_cast<REAL>(2) * pointsRadius[1] * static_cast<REAL>(rand()) / static_cast<REAL>(RAND_MAX) - pointsRadius[1];
            REAL rz = static_cast<REAL>(2) * pointsRadius[2] * static_cast<REAL>(rand()) / static_cast<REAL>(RAND_MAX) - pointsRadius[2];
            points3d.push_back(Vec3(pointsLocation[0] + rx, pointsLocation[1] + ry, pointsLocation[2] + rz));

            Track track;
            track.id = i;
            track.observations.reserve(cameraExtrinsics.size());
            tracks.push_back(track);
        }

        // This flips Y (Up -> Down) and Z (Look -Z -> Look +Z)
        Mat4 blenderToCv = Mat4::Identity();
        blenderToCv(1, 1) = -1;
        blenderToCv(2, 2) = -1;

        for (int i = 0; i < cameraExtrinsics.size(); i++)
        {
            Mat4 worldToBlender = cameraExtrinsics[i].inverse();
            Mat4 worldToCv = blenderToCv * worldToBlender;

            Mat4 poseInv = cameraExtrinsics[i].inverse();

            for (int j = 0; j < numPoints; j++)
            {
                Vec3 pos = cameraIntrinsics * (worldToCv * points3d[j].homogeneous()).head<3>();
                pos /= pos[2];

                REAL ru = static_cast<REAL>(2) * detectionError[0] * static_cast<REAL>(rand()) / static_cast<REAL>(RAND_MAX) - detectionError[0];
                REAL rv = static_cast<REAL>(2) * detectionError[1] * static_cast<REAL>(rand()) / static_cast<REAL>(RAND_MAX) - detectionError[1];

                Observation obs;
                obs.frameId = i;
                obs.point = Vec2(pos[0] + ru, pos[1] + rv);
                tracks[j].observations.push_back(obs);
            }
        }

        return tracks;
    }

    Mat4 calculateMatrix(REAL rotX, REAL rotY, REAL rotZ, Vec3 translation)
    {
        rotX = rotX * M_PI / (REAL)180;
        rotY = rotY * M_PI / (REAL)180;
        rotZ = rotZ * M_PI / (REAL)180;

        Mat3 Rx;
        Rx << (REAL)1, (REAL)0, (REAL)0,
            (REAL)0, std::cos(rotX), -std::sin(rotX),
            (REAL)0, std::sin(rotX), std::cos(rotX);

        Mat3 Ry;
        Ry << std::cos(rotY), (REAL)0, std::sin(rotY),
            (REAL)0, (REAL)1, (REAL)0,
            -std::sin(rotY), (REAL)0, std::cos(rotY);

        Mat3 Rz;
        Rz << std::cos(rotZ), -std::sin(rotZ), (REAL)0,
            std::sin(rotZ), std::cos(rotZ), (REAL)0,
            (REAL)0, (REAL)0, (REAL)1;

        Mat3 R = Rz * Ry * Rx;
        Mat4 res = Mat4::Identity();
        res.block<3, 3>(0, 0) = R;
        res.block<3, 1>(0, 3) = translation;
        return res;
    }
} // Namespace SfM::test