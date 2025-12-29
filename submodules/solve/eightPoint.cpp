#include "solve.hpp"
#include "../util/util.hpp"
#include <Eigen/SVD>
#include <iostream>
#include <unordered_set>

namespace SfM::solve
{
    /**
     * @brief Transforms a point from pixel coordinates to a range ~ -1...1
     */
    inline Vec2 normalizePoints(Vec2 pixel, Mat3 &K_inv)
    {
        Vec3 p_homog;
        p_homog << pixel[0], pixel[1], 1.0f;
        Vec3 ray = K_inv * p_homog;
        return Vec2(ray[0], ray[1]);
    }

    bool getObservationOrNull(Track &track, int frame, Vec2 &outObservation)
    {
        for (int i = track.lastIndex, max = track.observations.size(); i < max; i++)
        {
            if (track.observations[i].frameId == frame)
            {
                outObservation = track.observations[i].point;
                track.lastIndex = i;
                return true;
            }
        }
        /* for (const auto &observation : track.observations)
        {
            if (observation.frameId == frame)
            {
                outObservation = observation.point;
                return true;
            }
        } */
        return false;
    }

    REAL getMedian(std::vector<REAL> &v)
    {
        size_t n = v.size();
        if (n == 0)
            return 1.0;
        std::nth_element(v.begin(), v.begin() + n / 2, v.end());
        return v[n / 2];
    }

    SfMResult eightPointAlgorithm(std::vector<Frame> &frames, Mat3 K, const int numTotKeypoints)
    {
        Mat3 K_inv = K.inverse();

        std::vector<Vec2> shared12points1;
        std::vector<Vec2> shared12points2;

        std::vector<Vec2> shared23points2;
        std::vector<Vec2> shared23points3;

        std::vector<int> trackIndices12;
        std::vector<int> trackIndices23;

        shared12points1.reserve(numTotKeypoints);
        shared12points2.reserve(numTotKeypoints);

        shared23points2.reserve(numTotKeypoints);
        shared23points3.reserve(numTotKeypoints);

        trackIndices12.reserve(numTotKeypoints);
        trackIndices23.reserve(numTotKeypoints);

        Mat4 accumulatedPose = SfM::util::calculateTransformationMatrix(90, 0, 0, SfM::Vec3(0, 0, 0)); // Start Transform
        SfMResult result;
        result.points.resize(numTotKeypoints, Vec3::Zero());

        EightPointResult frame12;
        EightPointResult frame23;

        // First iteration outside the loop, then in the loop always add one more frame
        if (frames.size() < 2)
        {
            throw std::invalid_argument("Frames has to be at least of size 2 to solve for camera motion.");
        }
        int j = 0; // dont reset j every iteration because the keypoints are sorted by trackId
        for (auto &keypoint : frames[1].keypoints)
        {
            Vec2 o1;
            Vec2 o2;

            // Find the matching keypoint in the prevous frame
            if (keypoint.indexInLastFrame == Keypoint::UNINITIALIZED)
            {
                while (frames[0].keypoints[j].trackId <= keypoint.trackId && j < frames[0].keypoints.size())
                {
                    if (frames[0].keypoints[j].trackId == keypoint.trackId)
                    {
                        keypoint.indexInLastFrame = j;
                        j++;
                        break;
                    }
                    j++;
                }
                if (keypoint.indexInLastFrame == Keypoint::UNINITIALIZED)
                {
                    keypoint.indexInLastFrame = Keypoint::NOT_FOUND;
                }
            }

            if (keypoint.indexInLastFrame == Keypoint::NOT_FOUND)
            {
                continue;
            }
            o1 = frames[0].keypoints[keypoint.indexInLastFrame].point;
            o2 = keypoint.point;
            shared23points2.push_back(normalizePoints(o1, K_inv)); // use 23 here because it is the "last" for the triple -1, 0, 1 and in the loop we set 23 to 12
            shared23points3.push_back(normalizePoints(o2, K_inv));
            trackIndices23.push_back(keypoint.trackId);
        }

        frame23 = eightPointAlgorithm(shared23points2, shared23points3);

        for (int j = 0; j < frame23.points.size(); j++)
        {
            if (result.points[trackIndices23[j]] != Vec3::Zero())
            {
                result.points[trackIndices23[j]] = (util::blendCvMat() * accumulatedPose * frame23.points[j].homogeneous()).head<3>();
            }
        }

        result.extrinsics.push_back(accumulatedPose);
        result.extrinsics.push_back(accumulatedPose *= frame23.pose.inverse());

        // Add all the remaining frames one by one
        REAL scaleFactor = static_cast<REAL>(1);
        for (int i = 2, max = frames.size(); i < max; i++)
        {
            shared12points1 = std::move(shared23points2);
            shared12points2 = std::move(shared23points3);
            trackIndices12 = std::move(trackIndices23);
            frame12 = std::move(frame23);

            shared23points2.clear();
            shared23points3.clear();
            trackIndices23.clear();

            int j = 0;
            int idxInFrame12 = 0;
            for (auto &keypoint : frames[i].keypoints)
            {
                Vec2 o1;
                Vec2 o2;

                // Find the matching keypoint in the prevous frame
                if (keypoint.indexInLastFrame == Keypoint::UNINITIALIZED)
                {
                    while (frames[i - 1].keypoints[j].trackId <= keypoint.trackId && j < frames[i - 1].keypoints.size())
                    {
                        if (frames[i - 1].keypoints[j].trackId == keypoint.trackId)
                        {
                            keypoint.indexInLastFrame = j;
                            j++;
                            break;
                        }
                        j++;
                    }
                    if (keypoint.indexInLastFrame == Keypoint::UNINITIALIZED)
                    {
                        keypoint.indexInLastFrame = Keypoint::NOT_FOUND;
                    }
                }

                // Fill the array for the 8 point algorithm
                if (keypoint.indexInLastFrame == Keypoint::NOT_FOUND)
                {
                    continue;
                }

                const auto &matchInLastFrame = frames[i - 1].keypoints[keypoint.indexInLastFrame];
                o1 = matchInLastFrame.point;
                o2 = keypoint.point;
                shared23points2.push_back(normalizePoints(o1, K_inv)); // use 23 here because it is the "last" for the triple -1, 0, 1 and in the loop we set 23 to 12
                shared23points3.push_back(normalizePoints(o2, K_inv));
                trackIndices23.push_back(keypoint.trackId);
            }

            // Calculate new pose and points
            frame23 = eightPointAlgorithm(shared23points2, shared23points3);

            // Calculate scale between previous frame and current frame
            std::vector<REAL> ratios;
            int idx12 = 0;
            for (int idx23 = 0; idx23 < trackIndices23.size(); idx23++)
            {
                bool inAllThree = false;
                while (trackIndices12[idx12] <= trackIndices23[idx23] && idx12 < trackIndices12.size())
                {
                    if (trackIndices12[idx12] == trackIndices23[idx23])
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

                Vec3 match12Cam1 = frame12.points[trackIndices12[idx12 - 1]]; // -1 bc idx12 is always incremented
                Vec3 match23Cam2 = frame23.points[trackIndices23[idx23]];

                Vec3 match12Cam2 = (frame12.pose * match12Cam1.homogeneous()).head<3>();

                REAL dist12 = match12Cam2.norm();
                REAL dist23 = match23Cam2.norm();

                if (dist23 > EPSILON)
                {
                    ratios.push_back(dist12 / dist23);
                }
            }

            scaleFactor *= getMedian(ratios);
            std::cout << "Matching Frame" << i - 1 << i << " and Frame" << i << i + 1 << ", Accumulated Scale: " << scaleFactor << std::endl;

            // Add new pose and points to result
            Mat4 viewMat = frame23.pose;
            viewMat.block<3, 1>(0, 3) *= scaleFactor;

            for (int j = 0; j < frame23.points.size(); j++)
            {
                if (result.points[trackIndices23[j]] != Vec3::Zero())
                {
                    result.points[trackIndices23[j]] = (util::blendCvMat() * accumulatedPose * frame23.points[j].homogeneous()).head<3>();
                }
            }

            result.extrinsics.push_back(accumulatedPose *= viewMat.inverse());
        }

        return result;
    }

    SfMResult eightPointAlgorithm(std::vector<Track> &tracks, Mat3 K, const int numFrames)
    {
        Mat3 K_inv = K.inverse();

        std::vector<Vec2> shared12points1;
        std::vector<Vec2> shared12points2;

        std::vector<Vec2> shared23points2;
        std::vector<Vec2> shared23points3;

        std::vector<std::tuple<int, int>> shared123indices;

        std::vector<int> trackIndices;
        std::unordered_set<int> foundTracks;

        shared12points1.reserve(tracks.size());
        shared12points2.reserve(tracks.size());

        shared23points2.reserve(tracks.size());
        shared23points3.reserve(tracks.size());

        shared123indices.reserve(tracks.size());

        trackIndices.reserve(tracks.size());

        Mat4 accumulatedPose = SfM::util::calculateTransformationMatrix(90, 0, 0, SfM::Vec3(0, 0, 0)); // Start Transform
        SfMResult result;

        EightPointResult frame12;
        EightPointResult frame23;

        // First iteration outside the loop, then in the loop always add one more frame
        for (Track &t : tracks)
        {
            t.lastIndex = 0; // reset here

            Vec2 o1;
            Vec2 o2;

            if (getObservationOrNull(t, 0, o1) && getObservationOrNull(t, 1, o2))
            {
                shared23points2.push_back(normalizePoints(o1, K_inv)); // use 23 here because it is the "last" for the triple -1, 0, 1 and in the loop we set 23 to 12
                shared23points3.push_back(normalizePoints(o2, K_inv));
                trackIndices.push_back(t.id);
            };
        }

        frame23 = eightPointAlgorithm(shared23points2, shared23points3);

        for (int j = 0; j < frame23.points.size(); j++)
        {
            result.points.push_back((util::blendCvMat() * accumulatedPose * frame23.points[j].homogeneous()).head<3>());
            foundTracks.insert(trackIndices[j]);
        }

        result.extrinsics.push_back(accumulatedPose);
        result.extrinsics.push_back(accumulatedPose *= frame23.pose.inverse());

        // Add all the remaining frames one by one
        REAL scaleFactor = static_cast<REAL>(1);
        for (int i = 1; i < numFrames - 1; i++)
        {
            shared12points1 = std::move(shared23points2);
            shared12points2 = std::move(shared23points3);
            frame12 = std::move(frame23);

            shared23points2.clear();
            shared23points3.clear();
            shared123indices.clear();
            trackIndices.clear();

            // Find tracks with correspondences over the frames
            int idxInFrame12 = 0;
            for (Track &t : tracks)
            {
                Vec2 o2, o3;

                t.lastIndex = std::max(0, t.lastIndex - 2); // set back because we search for a frame we already wanted before
                bool inFrame12 = getObservationOrNull(t, i - 1, o2) && getObservationOrNull(t, i, o3);

                bool inFrame23 = false;
                if (getObservationOrNull(t, i, o2) && getObservationOrNull(t, i + 1, o3))
                {
                    inFrame23 = true;
                    shared23points2.push_back(normalizePoints(o2, K_inv));
                    shared23points3.push_back(normalizePoints(o3, K_inv));
                    trackIndices.push_back(t.id);
                }

                if (inFrame12 && inFrame23)
                {
                    shared123indices.push_back(std::make_tuple(idxInFrame12, shared23points2.size() - 1));
                }

                if (inFrame12)
                {
                    idxInFrame12++;
                }
            }

            // Calculate new pose and points
            frame23 = eightPointAlgorithm(shared23points2, shared23points3);

            // Calculate scale between previous frame and current frame
            std::vector<REAL> ratios;
            for (int j = 0; j < shared123indices.size(); j++)
            {
                Vec3 match12Cam1 = frame12.points[std::get<0>(shared123indices[j])];
                Vec3 match23Cam2 = frame23.points[std::get<1>(shared123indices[j])];

                Vec3 match12Cam2 = (frame12.pose * match12Cam1.homogeneous()).head<3>();

                REAL dist12 = match12Cam2.norm();
                REAL dist23 = match23Cam2.norm();

                if (dist23 > EPSILON)
                {
                    ratios.push_back(dist12 / dist23);
                }
            }

            scaleFactor *= getMedian(ratios);
            std::cout << "Matching Frame" << i - 1 << i << " and Frame" << i << i + 1 << ", Accumulated Scale: " << scaleFactor << ", shard point length: " << shared123indices.size() << std::endl;

            // Add new pose and points to result
            Mat4 viewMat = frame23.pose;
            viewMat.block<3, 1>(0, 3) *= scaleFactor;

            for (int j = 0; j < frame23.points.size(); j++)
            {
                if (foundTracks.find(trackIndices[j]) == foundTracks.end()) // Track not jet in the found tracks
                {
                    result.points.push_back((util::blendCvMat() * accumulatedPose * (scaleFactor * frame23.points[j]).homogeneous()).head<3>()); // use accumulated pose for last frame bc points are in Cam2 Space
                    foundTracks.insert(trackIndices[j]);
                }
            }

            result.extrinsics.push_back(accumulatedPose *= viewMat.inverse());
        }

        return result;
    }

    EightPointResult eightPointAlgorithm(const std::vector<Vec2> &points1, const std::vector<Vec2> &points2)
    {
        if (points1.size() < 8)
        {
            throw std::invalid_argument("There have to be at least 8 pair-to-pair-correspondences (was: " + std::to_string(points1.size()) + ") to approximate a pose.");
        }

        // row major indexing: matrix(row, column)
        const int n = points1.size();
        MatX A(n, 9);
        for (int i = 0; i < n; i++)
        {
            Vec3 x1;
            x1 << points1[i], static_cast<REAL>(1.);
            Vec3 x2;
            x2 << points2[i], static_cast<REAL>(1.);

            A(i, 0) = x1[0] * x2[0];
            A(i, 1) = x1[1] * x2[0];
            A(i, 2) = x1[2] * x2[0];
            A(i, 3) = x1[0] * x2[1];
            A(i, 4) = x1[1] * x2[1];
            A(i, 5) = x1[2] * x2[1];
            A(i, 6) = x1[0] * x2[2];
            A(i, 7) = x1[1] * x2[2];
            A(i, 8) = x1[2] * x2[2];
        }

        Eigen::JacobiSVD<MatX> svd = A.jacobiSvd(Eigen::ComputeThinV);
        VecX solution = svd.matrixV().col(8);

        Mat3 E_raw;
        E_raw.row(0) = solution.segment<3>(0);
        E_raw.row(1) = solution.segment<3>(3);
        E_raw.row(2) = solution.segment<3>(6);

        Eigen::JacobiSVD<Mat3> eSvd(E_raw, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Mat3 U = eSvd.matrixU();
        Vec3 S = eSvd.singularValues();
        Mat3 Vt = eSvd.matrixV().transpose();

        REAL sigma = (S[0] + S[1]) / static_cast<REAL>(2);
        Mat3 optimalSigma = Mat3::Zero();
        optimalSigma(0, 0) = sigma;
        optimalSigma(1, 1) = sigma;

        // Mat3 e = U * optimalSigma * Vt;

        const Mat3 rzPlusHalfPi = (Mat3() << 0.f, -1.f, 0.f, 1.f, 0.f, 0.f, 0.f, 0.f, 1.f).finished();
        const Mat3 rzMinusHalfPi = (Mat3() << 0.f, 1.f, 0.f, -1.f, 0.f, 0.f, 0.f, 0.f, 1.f).finished();

        Mat3 R1 = U * rzPlusHalfPi * Vt;
        if (R1.determinant() < 0)
        {
            R1 = -R1;
        }

        Mat3 R2 = U * rzMinusHalfPi * Vt;
        if (R2.determinant() < 0)
        {
            R2 = -R2;
        }

        Vec3 T1_2 = U.col(2);

        struct Candidate
        {
            Mat3 R;
            Vec3 T;
        };
        std::vector<Candidate> candidates = {
            {R1, T1_2}, {R1, -T1_2}, {R2, T1_2}, {R2, -T1_2}};

        int maxIndex = -1;
        int maxPoints = 0;

        for (int c = 0; c < 4; c++)
        {
            const Candidate &candidate = candidates[c];
            int numPoints = 0;

            for (int i = 0; i < n; i++)
            {
                Vec3 x1;
                x1 << points1[i], static_cast<REAL>(1.);
                Vec3 x2;
                x2 << points2[i], static_cast<REAL>(1.);

                /* Vec3 Rx1 = R * x1;
                Eigen::Matrix<REAL, 3, 2> A;
                A.col(0) = -Rx1;
                A.col(1) = x2;
                Vec2 lambdas = (A.transpose() * A).inverse() * (A.transpose() * T); */

                // Linear Triangulation (Least Squares)
                // Solve: lambda1 * x1 - lambda2 * (R*x2) = t
                // Matrix form: [x1  -R*x2] * [l1; l2] = t

                Vec3 Rx1 = candidate.R * x1;

                Eigen::Matrix<REAL, 3, 2> A_tri;
                A_tri.col(0) = -Rx1;
                A_tri.col(1) = x2;

                // Solve overdetermined system A_tri * x = T using SVD or QR
                Vec2 lambdas = A_tri.colPivHouseholderQr().solve(candidate.T);

                float lambda1 = lambdas[0];
                float lambda2 = lambdas[1];

                if (lambdas[0] > 0 && lambdas[1] > 0)
                {
                    numPoints++;
                }
            }

            if (numPoints > maxPoints)
            {
                maxIndex = c;
                maxPoints = numPoints;
            }
        }

        Mat3 R = candidates[maxIndex].R;
        Vec3 T = candidates[maxIndex].T;

        Mat4 pose = Mat4::Identity();
        pose.block<3, 3>(0, 0) = R;
        pose.block<3, 1>(0, 3) = T;

        std::vector<Vec3> points;
        points.reserve(n);

        for (int i = 0; i < n; i++)
        {
            Vec3 x1;
            x1 << points1[i], static_cast<REAL>(1.);
            Vec3 x2;
            x2 << points2[i], static_cast<REAL>(1.);

            /* Vec3 Rx1 = R * x1;
            Eigen::Matrix<REAL, 3, 2> A;
            A.col(0) = -Rx1;
            A.col(1) = x2;
            Vec2 lambdas = (A.transpose() * A).inverse() * (A.transpose() * T); */

            // Linear Triangulation (Least Squares)
            // Solve: lambda1 * x1 - lambda2 * (R*x2) = t
            // Matrix form: [x1  -R*x2] * [l1; l2] = t

            Vec3 Rx1 = R * x1;

            Eigen::Matrix<REAL, 3, 2> A_tri;
            A_tri.col(0) = -Rx1;
            A_tri.col(1) = x2;

            // Solve overdetermined system A_tri * x = T using SVD or QR
            Vec2 lambdas = A_tri.colPivHouseholderQr().solve(T);

            float lambda1 = lambdas[0];
            float lambda2 = lambdas[1];

            points.push_back(lambda1 * x1);
        }

        return {pose, points};
    };
} // Namespace SfM::solve