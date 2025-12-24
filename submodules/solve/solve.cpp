#include "solve.hpp"
#include "../util/util.hpp"
#include <Eigen/SVD>
#include <iostream>

namespace SfM::solve
{
    /**
     * @brief Transforms a point from pixel coordinates to a range ~ -1...1
     */
    /* inline Vec2 normalizePointsOld(Vec2 point, REAL widht, REAL height)
    {
        REAL max = std::max(widht, height);
        point[0] = (2 * point[0] - widht) / max;
        point[1] = (2 * point[1] - height) / max;
        return point;
    } */
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

    SfMResult eightPointAlgorithm(std::vector<Track> &tracks, Mat3 K, const int numFrames)
    {
        Mat3 K_inv = K.inverse();

        std::vector<Vec2> shared12points1;
        std::vector<Vec2> shared12points2;

        std::vector<Vec2> shared23points2;
        std::vector<Vec2> shared23points3;

        std::vector<std::tuple<int, int>> shared123indices;

        shared12points1.reserve(numFrames);
        shared12points2.reserve(numFrames);

        shared23points2.reserve(numFrames);
        shared23points3.reserve(numFrames);

        shared123indices.reserve(numFrames);

        Mat4 startTransform = SfM::util::calculateTransformationMatrix(90, 0, 0, SfM::Vec3(0, 0, 0));
        Mat4 accumulatedPose = startTransform;
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
            };
        }

        frame23 = eightPointAlgorithm(shared23points2, shared23points3);

        result.extrinsics.push_back(accumulatedPose);
        result.extrinsics.push_back(accumulatedPose *= frame23.pose.inverse());

        for (auto &point : frame23.points)
        {
            result.points.push_back((startTransform * util::blendCvMat() * point.homogeneous()).head<3>());
        }

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

            frame23 = eightPointAlgorithm(shared23points2, shared23points3);

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

            std::cout << "Matching Frame" << i - 1 << i << " and Frame" << i << i + 1 << ", Calculated Relative Scale: " << scaleFactor << std::endl;
            std::cout << "shared point length: " << shared123indices.size() << std::endl;

            // frame23.pose.block<3, 1>(0, 3) *= scaleFactor;

            Mat4 viewMat = frame23.pose;
            viewMat.block<3, 1>(0, 3) *= scaleFactor;

            result.extrinsics.push_back(accumulatedPose *= viewMat.inverse());

            /* for (auto &point : frame12.points)
            {
                point = (startTransform * util::blendCvMat() * point.homogeneous()).head<3>();
            } */
        }

        return result;
    }

    EightPointResult eightPointAlgorithm(const std::vector<Vec2> &points1, const std::vector<Vec2> &points2)
    {
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