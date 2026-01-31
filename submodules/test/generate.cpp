#include "generate.hpp"
#include <iostream>
#include <math.h>
#include "../util/util.hpp"

namespace SfM::test
{
    std::vector<Track> generateRandomPointsTracks(std::vector<Mat4> &cameraExtrinsics,
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

        for (int i = 0; i < cameraExtrinsics.size(); i++)
        {
            Mat4 worldToBlender = cameraExtrinsics[i].inverse();
            Mat4 worldToCv = util::blendCvMat4() * worldToBlender;

            for (int j = 0; j < numPoints; j++)
            {
                Vec3 pos = cameraIntrinsics * (worldToCv * points3d[j].homogeneous()).head<3>();
                pos /= pos[2];

                REAL ru = static_cast<REAL>(2) * detectionError[0] * static_cast<REAL>(rand()) / static_cast<REAL>(RAND_MAX) - detectionError[0];
                REAL rv = static_cast<REAL>(2) * detectionError[1] * static_cast<REAL>(rand()) / static_cast<REAL>(RAND_MAX) - detectionError[1];

                SimpleObservation obs;
                obs.frameId = i;
                obs.point = Vec2(pos[0] + ru, pos[1] + rv);

                if (static_cast<REAL>(rand()) / static_cast<REAL>(RAND_MAX) < 0.85)
                {
                    tracks[j].observations.push_back(obs);
                }
            }
        }

        return tracks;
    }

    inline REAL testReprojectionError(const Mat3 &K, const Vec2 observation, const Vec3 point3d, const Mat4 &viewMatrix)
    {
        Vec3 p = K * (viewMatrix * point3d.homogeneous()).head<3>();
        p[0] /= p[2];
        p[1] /= p[2];
        Vec3 obs(observation[0], observation[1], 1.);
        REAL dx = p[0] - obs[0];
        REAL dy = p[1] - obs[1];
        return dx * dx + dy * dy;
    }

    std::vector<Frame> generateRandomPointsFrames(std::vector<Mat4> &cameraExtrinsics,
                                                  Mat3 cameraIntrinsics,
                                                  Vec3 pointsLocation,
                                                  Vec3 pointsRadius,
                                                  std::optional<std::reference_wrapper<std::vector<Vec3>>> points,
                                                  int numPoints,
                                                  REAL inclusionProb,
                                                  Vec2 detectionError)
    {
        std::vector<Frame> frames;
        frames.reserve(cameraExtrinsics.size());

        std::vector<Vec3> ps{};
        std::vector<Vec3> &points3d = points ? points->get() : ps;
        points3d.reserve(numPoints);

        for (int i = 0; i < numPoints; i++)
        {
            REAL rx = static_cast<REAL>(2) * pointsRadius[0] * static_cast<REAL>(rand()) / static_cast<REAL>(RAND_MAX) - pointsRadius[0];
            REAL ry = static_cast<REAL>(2) * pointsRadius[1] * static_cast<REAL>(rand()) / static_cast<REAL>(RAND_MAX) - pointsRadius[1];
            REAL rz = static_cast<REAL>(2) * pointsRadius[2] * static_cast<REAL>(rand()) / static_cast<REAL>(RAND_MAX) - pointsRadius[2];
            points3d.push_back(Vec3(pointsLocation[0] + rx, pointsLocation[1] + ry, pointsLocation[2] + rz));
        }

        for (int i = 0; i < cameraExtrinsics.size(); i++)
        {
            Frame currentFrame;
            currentFrame.frameId = i;

            Mat4 worldToBlender = cameraExtrinsics[i].inverse();
            Mat4 worldToCv = util::blendCvMat4() * worldToBlender;

            for (int j = 0; j < numPoints; j++)
            {
                Vec3 pos = cameraIntrinsics * (worldToCv * points3d[j].homogeneous()).head<3>();
                pos /= pos[2];

                REAL ru = static_cast<REAL>(2) * detectionError[0] * static_cast<REAL>(rand()) / static_cast<REAL>(RAND_MAX) - detectionError[0];
                REAL rv = static_cast<REAL>(2) * detectionError[1] * static_cast<REAL>(rand()) / static_cast<REAL>(RAND_MAX) - detectionError[1];

                Observation observation;
                observation.trackId = j;
                observation.point = Vec2(pos[0] + ru, pos[1] + rv);

                if (static_cast<REAL>(rand()) / static_cast<REAL>(RAND_MAX) < inclusionProb)
                {
                    // std::cout << "observation at: " << observation.point[0] << ", " << observation.point[1] << std::endl;
                    // std::cout << "GENERATE: error " << testReprojectionError(cameraIntrinsics, observation.point, points3d[j], worldToCv) << std::endl;
                    currentFrame.observations.push_back(std::make_unique<Observation>(observation));
                }
            }

            frames.push_back(std::move(currentFrame));
        }

        return frames;
    }

    int addOutliersToFrames(std::vector<Frame> &frames, int minNumOutlierPerFrame, int maxNumOutlierPerFrame, int outliersStartIndex)
    {
        int maxAdded = 0;
        for (auto &frame : frames)
        {
            int spread = maxNumOutlierPerFrame - minNumOutlierPerFrame;
            int numOutliers = minNumOutlierPerFrame + static_cast<int>((static_cast<REAL>(rand()) / static_cast<REAL>(RAND_MAX) * (spread + 1)));
            maxAdded = std::max(maxAdded, numOutliers);

            std::cout << "added " << numOutliers << " outliers to frame" << std::endl;

            for (int i = 0; i < numOutliers; i++)
            {
                REAL u = static_cast<REAL>(1920) * static_cast<REAL>(rand()) / static_cast<REAL>(RAND_MAX);
                REAL v = static_cast<REAL>(1080) * static_cast<REAL>(rand()) / static_cast<REAL>(RAND_MAX);

                // std::cout << "outlier at: " << u << ", " << v << ";" << std::endl;

                Observation observation;
                observation.trackId = outliersStartIndex + i;
                observation.point = Vec2(u, v);

                frame.observations.push_back(std::make_unique<Observation>(observation));
            }
        }

        return maxAdded;
    }
} // Namespace SfM::test