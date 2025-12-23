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

    SfMResult eightPointAlgorithm(const std::vector<Track> &tracks, Mat3 K)
    {
        std::vector<Vec2> points1;
        std::vector<Vec2> points2;

        Mat3 K_inv = K.inverse();

        for (const auto &t : tracks)
        {
            Vec2 p1 = normalizePoints(t.observations[0].point, K_inv);
            Vec2 p2 = normalizePoints(t.observations[1].point, K_inv);

            points1.push_back(p1);
            points2.push_back(p2);
        }

        EightPointResult frame12 = eightPointAlgorithm(points1, points2);

        Mat4 start = util::blenderToCv(util::calculateTransformationMatrix(90, 0, 0, SfM::Vec3(0, 0, 0)));
        Mat4 start2 = util::blenderToCv(util::calculateTransformationMatrix(90, 0, 0, SfM::Vec3(0, 0, 0)));
        //start = Mat4::Identity();

        SfMResult result;
        result.extrinsics.push_back(start.inverse());
        result.extrinsics.push_back(start2.inverse() * frame12.pose.inverse());

        for (auto &point : frame12.points)
        {
            point = (start * point.homogeneous()).head<3>();
        }

        result.points = frame12.points;

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