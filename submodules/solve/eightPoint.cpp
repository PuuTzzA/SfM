#include "solve.hpp"
#include "../scene.hpp"
#include "../util/util.hpp"
#include <Eigen/SVD>
#include <iostream>
#include <unordered_set>

namespace SfM::solve
{
    SfMResult eightPointAlgorithm(std::vector<Frame> &frames, const Mat3 K, const int numTotTracks, const Mat4 startTransform)
    {
        SfM::Scene scene;
        scene.setK(K);
        scene.setStartTransform(startTransform);
        scene.setUseRANSAC(true);

        RANSAC_OPTIONS RANSAC_options;
        RANSAC_options.minN = 8;
        RANSAC_options.maxIter = 512;
        RANSAC_options.maxTimeMs = 1500;
        RANSAC_options.maxSquaredError = 10;
        RANSAC_options.successProb = 0.99;
        RANSAC_options.model = eightPointAlgorithmFromSubset;
        RANSAC_options.loss = [](const Mat3 &K, const Vec2 obs1, const Vec2 obs2, const Vec3 point3d, const Mat4 &viewMat)
        {
            REAL loss1 = reprojectionError(K, obs1, point3d, Mat4::Identity()); // is always ~0 since points are created with obs1 * lamba
            REAL loss2 = reprojectionError(K, obs2, point3d, viewMat);
            return std::max(loss1, loss2);
        };
        /* RANSAC_options.loss = [](const Mat3 &K, const Vec2 obs1, const Vec2 obs2, const Vec3 point3d, const Mat4 &viewMat)
        {
            return eightPointError(viewMat, obs1, obs2);
        }; */

        scene.setRANSACOptions(RANSAC_options);

        if (frames.size() < 2)
        {
            std::cerr << "frames has to be of size >= 2 to solve for camera motion!" << std::endl;
        }

        for (int i = 0; i < frames.size(); i++)
        {
            scene.addFrameWithoutMatching(std::move(frames[i]), numTotTracks);
        }

        return {scene.getExtrinsics(), scene.get3dPoints()};
    }

    EightPointResult eightPointAlgorithm(const std::vector<Vec2> &points1, const std::vector<Vec2> &points2)
    {
        if (points1.size() < 8)
        {
            throw std::invalid_argument("EightPointAlgorithm: There have to be at least 8 pair-to-pair-correspondences (was: " + std::to_string(points1.size()) + ") to approximate a pose.");
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

        Eigen::JacobiSVD<MatX> svd = A.jacobiSvd(Eigen::ComputeFullV);
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

        int maxIndex = 0;
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

    EightPointResult eightPointAlgorithmFromSubset(const std::vector<Vec2> &calcPoints1, const std::vector<Vec2> &calcPoints2,
                                                   const std::vector<Vec2> &allPoints1, const std::vector<Vec2> &allPoints2)
    {
        if (calcPoints1.size() < 8)
        {
            throw std::invalid_argument("EightPointFromSubset: There have to be at least 8 pair-to-pair-correspondences (was: " + std::to_string(calcPoints1.size()) + ") to approximate a pose.");
        }

        // row major indexing: matrix(row, column)
        const int n = calcPoints1.size();
        MatX A(n, 9);
        for (int i = 0; i < n; i++)
        {
            Vec3 x1;
            x1 << calcPoints1[i], static_cast<REAL>(1.);
            Vec3 x2;
            x2 << calcPoints2[i], static_cast<REAL>(1.);

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

        Eigen::JacobiSVD<MatX> svd = A.jacobiSvd(Eigen::ComputeFullV);
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

        int maxIndex = 0;
        int maxPoints = 0;

        for (int c = 0; c < 4; c++)
        {
            const Candidate &candidate = candidates[c];
            int numPoints = 0;

            for (int i = 0; i < n; i++)
            {
                Vec3 x1;
                x1 << calcPoints1[i], static_cast<REAL>(1.);
                Vec3 x2;
                x2 << calcPoints2[i], static_cast<REAL>(1.);

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
        points.reserve(allPoints1.size());

        for (int i = 0; i < allPoints1.size(); i++)
        {
            Vec3 x1;
            x1 << allPoints1[i], static_cast<REAL>(1.);
            Vec3 x2;
            x2 << allPoints2[i], static_cast<REAL>(1.);

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
    }
} // Namespace SfM::solve