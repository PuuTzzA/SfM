#include "scene.hpp"
#include "./util/util.hpp"
#include <Eigen/SVD>
#include <iostream>
#include <unordered_set>
#include <algorithm>

namespace SfM
{
    Scene::Scene(const Mat3 K, const Mat4 startTransform, SCENE_OPTIONS sceneOptions)
        : m_K{K}, m_K_inv{K.inverse()}, m_accumulatedPose{startTransform}, m_sceneOptions{sceneOptions} {};

    void Scene::setK(const Mat3 K)
    {
        m_K = K;
        m_K_inv = K.inverse();
    };

    void Scene::setStartTransform(const Mat4 startTransform)
    {
        m_accumulatedPose = startTransform;
    };

    void Scene::setSceneOptions(SCENE_OPTIONS sceneOptions)
    {
        m_sceneOptions = sceneOptions;
    }

    void Scene::setMatchingOptions(match::MATCHING_OPTIONS matchingOptions)
    {
        m_sceneOptions.matchingOptions = matchingOptions;
    }

    void Scene::setUseRANSAC(bool useRANSAC)
    {
        m_sceneOptions.useRANSAC = useRANSAC;
    };

    void Scene::setRANSACOptions(solve::RANSAC_OPTIONS ransacOptions)
    {
        m_sceneOptions.ransacOptions = ransacOptions;
    };

    void Scene::setBundleAdjustmentOptions(solve::BUNDLE_ADJUSTMENT_OPTIONS bundleAdjustmentOptions)
    {
        m_sceneOptions.bundleAdjustmentOptions = bundleAdjustmentOptions;
    }

    void Scene::pushBackImageWithKeypoints(cv::Mat &&image, std::vector<Keypoint> &&keypoints)
    {
        m_images.push_back(image);
        m_keypoints.push_back(keypoints);

        int lastIndex = m_keypoints.size() - 1;
        if (lastIndex < 1) // can only match if there are at least two images
        {
            return;
        }

        auto matches = match::match(m_keypoints[lastIndex - 1], m_keypoints[lastIndex], m_sceneOptions.matchingOptions);

        if (m_sceneOptions.verbose)
        {
            std::cout << "Scene::pushBackImageWithKeypoints: Matched: " << matches.size() << " keyframes between frame " << lastIndex - 1 << " and " << lastIndex << std::endl;
        }

        if (lastIndex == 1) // Create new frame
        {
            m_frames.push_back({.frameId = 0});
            m_frames.push_back({.frameId = 1});
        }
        else
        {
            m_frames.push_back({.frameId = lastIndex});
        }

        Frame &frameA = m_frames[lastIndex - 1];
        Frame &frameB = m_frames[lastIndex];

        for (const auto &[idxA, idxB] : matches)
        {
            Keypoint &kpA = m_keypoints[lastIndex - 1][idxA];
            Keypoint &kpB = m_keypoints[lastIndex][idxB];

            if (kpA.trackId == Keypoint::UNINITIALIZED) // new Track
            {
                kpA.trackId = m_currentNumTracks;
                kpB.trackId = m_currentNumTracks;

                frameA.observations.push_back(std::make_unique<Observation>(Observation{.point = kpA.point, .trackId = m_currentNumTracks}));
                kpA.observation = frameA.observations.back().get();
                frameB.observations.push_back(std::make_unique<Observation>(Observation{.point = kpB.point, .trackId = m_currentNumTracks}));
                kpB.observation = frameB.observations.back().get();

                m_currentNumTracks++;
            }
            else // existing Track. Add information to kbB
            {
                if (m_sceneOptions.splitTracks && kpA.observation != nullptr && !kpA.observation->inlier) // If the obervation was found to be an outlier, create a new TrackId
                {
                    kpA.observation->inlier = true; // reset inlier flag for the new track
                    kpA.observation->wasOutlierBefore = true;
                    kpA.observation->indexInLastFrame = Observation::UNINITIALIZED;
                    kpA.observation->trackId = m_currentNumTracks;

                    kpA.trackId = m_currentNumTracks;
                    kpB.trackId = m_currentNumTracks;

                    frameB.observations.push_back(std::make_unique<Observation>(Observation{.point = kpB.point, .trackId = m_currentNumTracks}));
                    kpB.observation = frameB.observations.back().get();

                    m_currentNumTracks++;
                }
                else
                {
                    kpB.trackId = kpA.trackId;
                    frameB.observations.push_back(std::make_unique<Observation>(Observation{.point = kpB.point, .trackId = kpA.trackId}));
                    kpB.observation = frameB.observations.back().get();
                }
            }
        };

        // util::drawCollageWithTracks({m_images[lastIndex - 1], m_images[lastIndex]}, tracks, 0, 2, "../../Data/found_tracks_" + std::to_string(lastIndex) + ".png");

        if (lastIndex == 1) // For the first two frames the observations are already in order
        {
            initializeEgithPointVariables();
        }
        else
        {
            std::sort(frameA.observations.begin(), frameA.observations.end(), [](const std::unique_ptr<Observation> &a, const std::unique_ptr<Observation> &b)
                      { return a->trackId < b->trackId; });

            std::sort(frameB.observations.begin(), frameB.observations.end(), [](const std::unique_ptr<Observation> &a, const std::unique_ptr<Observation> &b)
                      { return a->trackId < b->trackId; });
        }

        if (m_sceneOptions.useEightPoint)
        {
            solveForLastAddedFrame();
        }
    }

    void Scene::optimizeExtrinsicsAnd3dPoints()
    {
        SfMResult optimized;
        if (m_sceneOptions.useEightPoint)
        {
            const SfMResult initial{
                .extrinsics = std::move(m_extrinsics),
                .points = std::move(m_points3d),
            };
            optimized = solve::bundleAdjustment(m_frames, m_K, m_currentNumTracks, m_sceneOptions.bundleAdjustmentOptions, &initial, Mat4::Identity());
        }
        else
        {
            optimized = solve::bundleAdjustment(m_frames, m_K, m_currentNumTracks, m_sceneOptions.bundleAdjustmentOptions, nullptr, m_accumulatedPose);
        }
        m_extrinsics = std::move(optimized.extrinsics);
        m_points3d = std::move(optimized.points);
        m_points3dFilterd = std::move(optimized.pointsFiltered);
    }

    void Scene::initializeEgithPointVariables()
    {
        m_points3d.resize(m_currentNumTracks, Vec3::Zero());
        m_colors.resize(m_currentNumTracks, Vec3rgb::Zero());
        m_point3dCounts.resize(m_currentNumTracks, 0);
        m_extrinsics.push_back(m_accumulatedPose);
    }

    void Scene::solveForLastAddedFrame()
    {
        if (m_currentNumTracks < m_points3d.size())
        {
            std::cerr << "Scene::solveForLastAddedFrame: The number of 3d points should never decrease! was " << m_points3d.size() << ", got " << m_currentNumTracks << std::endl;
            return;
        }
        if (m_frames.size() < 2)
        {
            std::cerr << "Scene::solveForLastAddedFrame called but the number of frames was not >= 2, was: " << m_frames.size() << "!" << std::endl;
        }

        m_points3d.resize(m_currentNumTracks, Vec3::Zero());
        m_colors.resize(m_currentNumTracks, Vec3rgb::Zero());
        m_point3dCounts.resize(m_currentNumTracks, 0);
        int n = m_frames.size() - 1;

        // Reset and move points
        m_shared12points1 = std::move(m_shared23points2);
        m_shared12points2 = std::move(m_shared23points3);
        m_trackIndices12 = std::move(m_trackIndices23);
        m_frame12 = std::move(m_frame23);

        m_shared23points2.clear();
        m_shared23points3.clear();
        m_trackIndices23.clear();

        std::vector<int> indicesOfNewerObservations; // used to find the observations in frame.observations to set the inlier flag to 0
        std::vector<int> indicesOfOlderObservations;

        int j = 0; // dont reset j every iteration because the observations are sorted by trackId
        // for (auto &observation : m_frames[n].observations)
        for (int i = 0, max = m_frames[n].observations.size(); i < max; i++)
        {
            Observation &observation = *m_frames[n].observations[i];

            // Find the matching observation in the prevous frame
            if (observation.indexInLastFrame == Observation::UNINITIALIZED)
            {
                while (j < m_frames[n - 1].observations.size() && m_frames[n - 1].observations[j]->trackId <= observation.trackId)
                {
                    if (m_frames[n - 1].observations[j]->trackId == observation.trackId)
                    {
                        observation.indexInLastFrame = j;
                        j++;
                        break;
                    }
                    j++;
                }
                if (observation.indexInLastFrame == Observation::UNINITIALIZED)
                {
                    observation.indexInLastFrame = Observation::NOT_FOUND;
                }
            }

            if (observation.indexInLastFrame == Observation::NOT_FOUND)
            {
                continue;
            }

            Vec2 o1 = m_frames[n - 1].observations[observation.indexInLastFrame]->point;
            Vec2 o2 = observation.point;
            m_shared23points2.push_back(normalizePoints(o1));
            m_shared23points3.push_back(normalizePoints(o2));
            m_trackIndices23.push_back(observation.trackId);

            indicesOfNewerObservations.push_back(i);
            indicesOfOlderObservations.push_back(observation.indexInLastFrame);
        }

        // Calculate new pose and points
        if (!m_sceneOptions.useRANSAC)
        {
            m_frame23 = solve::eightPointAlgorithm(m_shared23points2, m_shared23points3);
        }
        else
        {
            auto inliers = solve::RANSAC(m_shared23points2, m_shared23points3, m_K, m_sceneOptions.ransacOptions, m_sceneOptions.verbose);
            std::sort(inliers.begin(), inliers.end());

            if (inliers.size() >= 8)
            {
                std::vector<Vec2> inliers1;
                std::vector<Vec2> inliers2;
                std::vector<int> inliersTrackIndeces;

                auto &newObs = m_frames[n].observations;
                auto &oldObs = m_frames[n - 1].observations;
                std::vector<bool> inlierMask(indicesOfNewerObservations.size(), false);

                for (const auto &i : inliers)
                {
                    inlierMask[i] = true;
                    inliers1.push_back(m_shared23points2[i]);
                    inliers2.push_back(m_shared23points3[i]);
                    inliersTrackIndeces.push_back(m_trackIndices23[i]);
                }

                for (int i = 0, max = inlierMask.size(); i < max; i++)
                {
                    newObs[indicesOfNewerObservations[i]]->inlier = inlierMask[i];

                    if (oldObs[indicesOfOlderObservations[i]]->wasOutlierBefore && inlierMask[i] == false)
                    {
                        oldObs[indicesOfOlderObservations[i]]->inlier = false;
                    }
                }

                m_frame23 = solve::eightPointAlgorithm(inliers1, inliers2);

                m_shared23points2 = std::move(inliers1);
                m_shared23points3 = std::move(inliers2);
                m_trackIndices23 = std::move(inliersTrackIndeces);
            }
            else
            {
                std::cerr << "RANSAC failed to find 8 inliers. Using all points." << std::endl;
                m_frame23 = solve::eightPointAlgorithm(m_shared23points2, m_shared23points3);
            }
        }

        // If this is not the second frame, match the scale to the previous frames
        if (n != 1)
        {
            std::vector<REAL> ratios;
            int idx12 = 0;
            int numInAllThree = 0;
            for (int idx23 = 0; idx23 < m_trackIndices23.size(); idx23++)
            {
                int currentTrackId = m_trackIndices23[idx23];

                while (idx12 < m_trackIndices12.size() && m_trackIndices12[idx12] < currentTrackId)
                {
                    idx12++;
                }

                if (idx12 >= m_trackIndices12.size())
                {
                    break;
                }

                if (m_trackIndices12[idx12] == currentTrackId)
                {
                    numInAllThree++;

                    Vec3 pointCam1Frame12 = m_frame12.points[idx12];
                    Vec3 pointCam2Frame12 = (m_frame12.pose * pointCam1Frame12.homogeneous()).head<3>();

                    Vec3 pointCam2Frame23 = m_frame23.points[idx23];

                    REAL dist12 = pointCam2Frame12.norm();
                    REAL dist23 = pointCam2Frame23.norm();

                    if (dist23 > 0.1 && dist12 > 0.1 && dist23 < 100.0 && dist12 < 100.0)
                    {
                        ratios.push_back(dist12 / dist23);
                    }
                }
            }

            if (ratios.size() >= 5)
            {
                REAL relativeScale = getMedian(ratios);

                if (relativeScale >= 0.1 && relativeScale < 10)
                {
                    m_accumulatedScale *= relativeScale;
                }
            }

            if (m_sceneOptions.verbose)
            {
                std::cout << "Scene::solveForLastAddedFrame: Matching scale between frame " << n << " and previous frames, Accumulated Scale: " << m_accumulatedScale << ", Matching points: " << numInAllThree << std::endl;
            }
        }

        Mat4 viewMat = m_frame23.pose;
        viewMat.block<3, 1>(0, 3) *= m_accumulatedScale;

        REAL translationLength = viewMat.block<3, 1>(0, 3).norm();
        if (translationLength > m_sceneOptions.maxTranslationPerFrame)
        {
            std::cerr << "WARNING: Translation length exceeded length of " << m_sceneOptions.maxTranslationPerFrame << ", was: " << translationLength << "\n";
            std::cerr << "Rescaled translation to ||translation|| = " << m_sceneOptions.maxTranslationPerFrame << std::endl;
            viewMat.block<3, 1>(0, 3) *= (m_sceneOptions.maxTranslationPerFrame / translationLength);
        }

        for (int j = 0; j < m_frame23.points.size(); j++)
        {
            int trackId = m_trackIndices23[j];

            if (m_frame23.points[j][2] < 0) // point behind the camera, bad triangulation
            {
                continue;
            }

            Vec3 newPointGlobal = (m_accumulatedPose * (m_accumulatedScale * m_frame23.points[j]).homogeneous()).head<3>();
            Vec3rgb newPointColor = util::getPixelBilinearUchar(m_images[n - 1], denormalizePoints(m_shared23points2[j]));

            if (m_point3dCounts[trackId] == 0)
            {
                m_points3d[trackId] = newPointGlobal;
                m_colors[trackId] = newPointColor;
                m_point3dCounts[trackId] = 1;
            }
            else // Update 3d point using a running average
            {
                if ((m_points3d[trackId] - newPointGlobal).norm() < 50.0)
                {
                    int N = m_point3dCounts[trackId];

                    Vec3 oldPoint = m_points3d[trackId];
                    m_points3d[trackId] = oldPoint + (newPointGlobal - oldPoint) / static_cast<REAL>(N + 1);

                    Vec3rgb oldColor = m_colors[trackId];

                    auto accumulateUchar = [N](unsigned char cNew, unsigned char cOld) -> unsigned char {
                        float cNewf = static_cast<float>(cNew);
                        float cOldf = static_cast<float>(cOld);
                        float res = cOldf + (cNewf - cOldf) / static_cast<float>(N + 1);
                        res = std::clamp(res, 0.f, 255.f);
                        return static_cast<unsigned char>(res);
                    };

                    m_colors[trackId][0] = accumulateUchar(newPointColor[0], oldColor[0]);
                    m_colors[trackId][1] = accumulateUchar(newPointColor[1], oldColor[1]);
                    m_colors[trackId][2] = accumulateUchar(newPointColor[2], oldColor[2]);

                    m_point3dCounts[trackId]++;
                }
            }

            /* if (m_trackIndices23[j] == 0)
            {
                std::cout << "1 point with an error of: " << std::sqrt(reprojectionError(m_K, m_shared23points2[j], m_frame23.points[j], Mat4::Identity())) << std::endl;
                std::cout << "2 point with an error of: " << std::sqrt(reprojectionError(m_K, m_shared23points3[j], m_frame23.points[j], m_frame23.pose)) << std::endl;
                std::cout << "eightPointError of: " << eightPointError(m_frame23.pose, m_shared23points2[j], m_shared23points3[j]) << std::endl;
            } */
        }

        m_extrinsics.push_back(m_accumulatedPose *= viewMat.inverse());
    };

    void Scene::addFrameWithoutMatching(Frame &&frame, const int newNumTotTracks)
    {
        m_currentNumTracks = newNumTotTracks;
        m_frames.push_back(std::move(frame));

        if (m_frames.size() == 1)
        {
            initializeEgithPointVariables();
            return;
        }
        solveForLastAddedFrame();
    }

    Mat3 Scene::getK()
    {
        return m_K;
    }

    std::vector<cv::Mat> &Scene::getImages()
    {
        return m_images;
    }

    std::vector<Mat4> &Scene::getExtrinsics()
    {
        return m_extrinsics;
    };

    std::vector<Vec3> &Scene::get3dPoints()
    {
        return m_points3d;
    };

    std::vector<Vec3> &Scene::get3dPointsFilterd()
    {
        if (m_points3dFilterd.size() > 0)
        {
            return m_points3dFilterd;
        }
        for (const auto &p : m_points3d)
        {
            if (p != Vec3::Zero())
            {
                m_points3dFilterd.push_back(p);
            }
        }
        return m_points3dFilterd;
    };

    std::vector<Vec3rgb> &Scene::getColors()
    {
        return m_colors;
    }

    REAL Scene::getMedian(std::vector<REAL> &v)
    {
        size_t n = v.size();
        if (n == 0)
            return 1.0;
        std::nth_element(v.begin(), v.begin() + n / 2, v.end());
        return v[n / 2];
    }

    Vec2 Scene::normalizePoints(Vec2 pixel)
    {
        Vec3 p_homog;
        p_homog << pixel[0], pixel[1], 1.0f;
        Vec3 ray = m_K_inv * p_homog;
        return Vec2(ray[0], ray[1]);
    };

    Vec2 Scene::denormalizePoints(Vec2 normalizedPoint)
    {
        Vec3 ray;
        ray << normalizedPoint[0], normalizedPoint[1], 1.0f;
        Vec3 p_homog = m_K * ray;
        return Vec2(p_homog[0], p_homog[1]);
    };
} // Namespace SfM