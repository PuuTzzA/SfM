#include "generate.hpp"
#include <iostream>
#include <math.h>
#include <Eigen/Dense> 

namespace SfM::test
{
    std::vector<Track> generateRandomPoints(std::vector<Mat4> &cameraPoses,
                                            Mat4 cameraIntrinsics,
                                            Vec3 pointsLocation,
                                            Vec3 pointsRadius,
                                            std::optional<std::reference_wrapper<std::vector<Vec3>>> points,
                                            int numPoints,
                                            REAL detectionError)
    {
        std::vector<Track> tracks;
        tracks.reserve(numPoints);

        std::vector<Vec3> ps{};
        std::vector<Vec3> &points3d = points ? points->get() : ps;
        points3d.reserve(numPoints);

        for (int i = 0; i < numPoints; i++)
        {
            REAL rx = pointsRadius[0] * static_cast<REAL>(rand()) / static_cast<REAL>(RAND_MAX);
            REAL ry = pointsRadius[1] * static_cast<REAL>(rand()) / static_cast<REAL>(RAND_MAX);
            REAL rz = pointsRadius[2] * static_cast<REAL>(rand()) / static_cast<REAL>(RAND_MAX);
            points3d.push_back(Vec3(pointsLocation[0] + rx, pointsLocation[1] + ry, pointsLocation[2] + rz));

            Track track;
            track.id = i;
            track.observations.reserve(cameraPoses.size());
            tracks.push_back(track);
        }

        for (int i = 0; i < cameraPoses.size(); i++)
        {
            Mat4 poseInv = cameraPoses[i].inverse();

            for (int j = 0; j < numPoints; j++)
            {
                Vec4 pos;
                pos << points3d[j], static_cast<REAL>(1);
                pos = cameraIntrinsics * poseInv * pos;
                pos /= pos[3];

                REAL ru = detectionError * static_cast<REAL>(rand()) / static_cast<REAL>(RAND_MAX);
                REAL rv = detectionError * static_cast<REAL>(rand()) / static_cast<REAL>(RAND_MAX);

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