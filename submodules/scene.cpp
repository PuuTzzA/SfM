#include "scene.hpp"
#include "./util/util.hpp"
#include <Eigen/SVD>
#include <iostream>
#include <unordered_set>

namespace SfM
{
    Scene::Scene(const Mat3 K, const Mat4 startTransform, bool useRANSAC, solve::RANSAC_OPTIONS RANSAC_options)
        : m_K{K}, m_K_inv{K.inverse()}, m_accumulatedPose{startTransform}, m_useRANSAC{useRANSAC}, m_RANSAC_options{RANSAC_options} {};

    void Scene::setK(const Mat3 K)
    {
        m_K = K;
        m_K_inv = K.inverse();
    };

    void Scene::setStartTransform(const Mat4 startTransform)
    {
        m_accumulatedPose = startTransform;
    };

    void Scene::setUseRANSAC(bool useRANSAC)
    {
        m_useRANSAC = useRANSAC;
    };

    void Scene::setRANSACOptions(solve::RANSAC_OPTIONS RANSAC_options)
    {
        m_RANSAC_options = RANSAC_options;
    };

    void Scene::initializeFromFirstFrame(Frame &&frame, const int numTotTracks)
    {
        m_shared12points1.reserve(numTotTracks);
        m_shared12points2.reserve(numTotTracks);

        m_shared23points2.reserve(numTotTracks);
        m_shared23points3.reserve(numTotTracks);

        m_trackIndices12.reserve(numTotTracks);
        m_trackIndices23.reserve(numTotTracks);

        m_points3d.resize(numTotTracks, Vec3::Zero());
        m_extrinsics.push_back(m_accumulatedPose);

        m_frames.push_back(std::move(frame));
    };

    REAL getMedian(std::vector<REAL> &v)
    {
        size_t n = v.size();
        if (n == 0)
            return 1.0;
        std::nth_element(v.begin(), v.begin() + n / 2, v.end());
        return v[n / 2];
    }

    void Scene::addFrame(Frame &&frame, const int newNumTotTracks)
    {
        if (newNumTotTracks < m_points3d.size())
        {
            std::cerr << "The number of 3d points should never decrease! was " << m_points3d.size() << ", got " << newNumTotTracks << std::endl;
            return;
        }
        m_frames.push_back(std::move(frame));
        if (m_frames.size() < 2)
        {
            std::cerr << "Scene::initializeFromFirstFrame must be called before calling Scene::addFrame!" << std::endl;
        }

        m_points3d.resize(newNumTotTracks, Vec3::Zero());
        int n = m_frames.size() - 1;

        // Reset and move points
        m_shared12points1 = std::move(m_shared23points2);
        m_shared12points2 = std::move(m_shared23points3);
        m_trackIndices12 = std::move(m_trackIndices23);
        m_frame12 = std::move(m_frame23);

        m_shared23points2.clear();
        m_shared23points3.clear();
        m_trackIndices23.clear();

        int j = 0; // dont reset j every iteration because the observations are sorted by trackId
        for (auto &observation : m_frames[n].observations)
        {
            Vec2 o1;
            Vec2 o2;

            // Find the matching observation in the prevous frame
            if (observation.indexInLastFrame == Observation::UNINITIALIZED)
            {
                while (m_frames[n - 1].observations[j].trackId <= observation.trackId && j < m_frames[n - 1].observations.size())
                {
                    if (m_frames[n - 1].observations[j].trackId == observation.trackId)
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

            o1 = m_frames[n - 1].observations[observation.indexInLastFrame].point;
            o2 = observation.point;
            m_shared23points2.push_back(normalizePoints(o1));
            m_shared23points3.push_back(normalizePoints(o2));
            m_trackIndices23.push_back(observation.trackId);
        }

        // Calculate new pose and points
        if (!m_useRANSAC)
        {
            m_frame23 = solve::eightPointAlgorithm(m_shared23points2, m_shared23points3);
        }
        else
        {
            auto inliers = solve::RANSAC(m_shared23points2, m_shared23points3, m_K, m_RANSAC_options);

            if (inliers.size() >= 8)
            {
                std::vector<Vec2> inliers1;
                std::vector<Vec2> inliers2;
                std::vector<int> inliersTrackIndeces;

                for (const auto &i : inliers)
                {
                    inliers1.push_back(m_shared23points2[i]);
                    inliers2.push_back(m_shared23points3[i]);
                    inliersTrackIndeces.push_back(m_trackIndices23[i]);
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
                bool inAllThree = false;
                while (m_trackIndices12[idx12] <= m_trackIndices23[idx23] && idx12 < m_trackIndices12.size())
                {
                    if (m_trackIndices12[idx12] == m_trackIndices23[idx23])
                    {
                        inAllThree = true;
                        idx12++;
                        break;
                    }
                    idx12++;
                }

                if (!inAllThree)
                {
                    continue;
                }

                numInAllThree++;

                Vec3 match12Cam1 = m_frame12.points[idx12 - 1]; // minus one since idx12 is always incremented
                Vec3 match23Cam2 = m_frame23.points[idx23];     // idx23 is the current loop index

                Vec3 match12Cam2 = (m_frame12.pose * match12Cam1.homogeneous()).head<3>();

                REAL dist12 = match12Cam2.norm();
                REAL dist23 = match23Cam2.norm();

                if (dist23 > EPSILON)
                {
                    ratios.push_back(dist12 / dist23);
                }
            }

            m_accumulatedScale *= getMedian(ratios);
            std::cout << "Matching Frame" << n - 2 << n - 1 << " and Frame" << n - 1 << n << ", Accumulated Scale: " << m_accumulatedScale << ", Matching points: " << numInAllThree << std::endl;
        }

        Mat4 viewMat = m_frame23.pose;
        viewMat.block<3, 1>(0, 3) *= m_accumulatedScale;

        for (int j = 0; j < m_frame23.points.size(); j++)
        {
            /* if (m_trackIndices23[j] == 0)
            {
                std::cout << "1 point with an error of: " << std::sqrt(reprojectionError(m_K, m_shared23points2[j], m_frame23.points[j], Mat4::Identity())) << std::endl;
                std::cout << "2 point with an error of: " << std::sqrt(reprojectionError(m_K, m_shared23points3[j], m_frame23.points[j], m_frame23.pose)) << std::endl;
                std::cout << "eightPointError of: " << eightPointError(m_frame23.pose, m_shared23points2[j], m_shared23points3[j]) << std::endl;
            } */

            // Add the 3d point if it is new
            if (m_points3d[m_trackIndices23[j]] == Vec3::Zero())
            {
                m_points3d[m_trackIndices23[j]] = (m_accumulatedPose * (m_accumulatedScale * m_frame23.points[j]).homogeneous()).head<3>();
            }
        }

        m_extrinsics.push_back(m_accumulatedPose *= viewMat.inverse());
    };

    std::vector<Mat4> Scene::getExtrinsics()
    {
        return m_extrinsics;
    };

    std::vector<Vec3> Scene::get3dPoints()
    {
        return m_points3d;
    };

    Vec2 Scene::normalizePoints(Vec2 pixel)
    {
        Vec3 p_homog;
        p_homog << pixel[0], pixel[1], 1.0f;
        Vec3 ray = m_K_inv * p_homog;
        return Vec2(ray[0], ray[1]);
    };
} // Namespace SfM