#include <Eigen/Core>
#include <vector>
#include <Eigen/SVD>
#include <iostream>
#include <Eigen/Dense>
#include "solve.h"

namespace SfM::Solve
{
    SfM::SfMResult eightPointAlgorithm(std::vector<std::vector<Eigen::Vector2f>> tracks)
    {
        // Intrinsics matrix here for now (perfect blender for hd and focal lenght 50mm and sensor size 36mm)
        float width_px = 1920.f;
        float height_px = 1080.f;
        float f_mm = 50.f;
        float sensor_mm = 36.f;

        float fx = f_mm * width_px / sensor_mm; // focal length in pixel
        float fy = fx;
        float cx = width_px / 2.0f;  // 960
        float cy = height_px / 2.0f; // 540

        Eigen::Matrix3f K;
        K << fx, 0, cx,
            0, fy, cy,
            0, 0, 1;

        Eigen::Matrix3f K_inv = K.inverse();

        int startFrame = 0;
        // row major indexing: matrix(row, column)
        Eigen::MatrixXf A(tracks.size(), 9);
        for (int i = 0; i < tracks.size(); i++)
        {

            auto &track = tracks[i];

            auto x1 = track[startFrame].array() * Eigen::Vector2f(width_px, height_px).array();
            auto x2 = track[startFrame + 1].array() * Eigen::Vector2f(width_px, height_px).array();

            Eigen::Vector3f x1_norm = K_inv * Eigen::Vector3f(x1[0], x1[1], 1.);
            Eigen::Vector3f x2_norm = K_inv * Eigen::Vector3f(x2[0], x2[1], 1.);

            A(i, 0) = x1_norm[0] * x2_norm[0];
            A(i, 1) = x1_norm[1] * x2_norm[0];
            A(i, 2) = x1_norm[2] * x2_norm[0];
            A(i, 3) = x1_norm[0] * x2_norm[1];
            A(i, 4) = x1_norm[1] * x2_norm[1];
            A(i, 5) = x1_norm[2] * x2_norm[1];
            A(i, 6) = x1_norm[0] * x2_norm[2];
            A(i, 7) = x1_norm[1] * x2_norm[2];
            A(i, 8) = x1_norm[2] * x2_norm[2];
        }

        Eigen::JacobiSVD<Eigen::MatrixXf> svd = A.jacobiSvd(Eigen::ComputeThinV);

        Eigen::Matrix3f essentialMatrixRaw;
        Eigen::VectorXf solution = svd.matrixV().col(8);

        Eigen::Matrix3f E_raw;
        E_raw.row(0) = solution.segment<3>(0);
        E_raw.row(1) = solution.segment<3>(3);
        E_raw.row(2) = solution.segment<3>(6);

        Eigen::JacobiSVD<Eigen::Matrix3f> eSvd(E_raw, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::Matrix3f U = eSvd.matrixU();
        Eigen::Matrix3f Vt = eSvd.matrixV().transpose();

        // STRICTLY enforce positive determinants for U and Vt to avoid reflection issues
        if (U.determinant() < 0)
            U.col(2) *= -1;
        if (Vt.determinant() < 0)
            Vt.row(2) *= -1;

        const Eigen::Matrix3f rzPlusHalfPi = (Eigen::Matrix3f() << 0.f, -1.f, 0.f, 1.f, 0.f, 0.f, 0.f, 0.f, 1.f).finished();
        const Eigen::Matrix3f rzMinusHalfPi = (Eigen::Matrix3f() << 0.f, 1.f, 0.f, -1.f, 0.f, 0.f, 0.f, 0.f, 1.f).finished();
        const Eigen::Matrix3f optimalSigma = (Eigen::Matrix3f() << 1.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 0.f).finished();

        Eigen::Matrix3f E = U * optimalSigma * Vt;

        Eigen::Matrix3f R1 = U * rzPlusHalfPi * Vt;
        Eigen::Matrix3f R2 = U * rzMinusHalfPi * Vt;

        Eigen::Vector3f t = U.col(2).normalized(); // t is up to scale

        std::vector<std::pair<Eigen::Matrix3f, Eigen::Vector3f>> candidates = {
            {R1, t}, {R1, -t}, {R2, t}, {R2, -t}};

        // --- 6. Cheirality Check (Triangulation) ---
        int bestIdx = -1;
        int maxInFront = -1;
        std::vector<Eigen::Vector3f> bestPoints;

        for (int i = 0; i < 4; i++)
        {
            const auto &R = candidates[i].first;
            const auto &t = candidates[i].second;

            int inFront = 0;
            std::vector<Eigen::Vector3f> currentPoints;
            currentPoints.reserve(tracks.size());

            // Test each track point
            for (const auto &track : tracks)
            {
                Eigen::Vector2f px1 = track[startFrame].array() * Eigen::Vector2f(width_px, height_px).array();
                Eigen::Vector2f px2 = track[startFrame + 1].array() * Eigen::Vector2f(width_px, height_px).array();

                Eigen::Vector3f x1 = K_inv * Eigen::Vector3f(px1.x(), px1.y(), 1.0f);
                Eigen::Vector3f x2 = K_inv * Eigen::Vector3f(px2.x(), px2.y(), 1.0f);

                /* Eigen::Vector3f Rx1 = R * x1;

                Eigen::Matrix<float, 3, 2> A;
                A.col(0) = -Rx1;
                A.col(1) = x2;

                Eigen::Vector2f lambdas = (A.transpose() * A).inverse() * (A.transpose() * t); */

                // Linear Triangulation (Least Squares)
                // Solve: lambda1 * x1 - lambda2 * (R*x2) = t
                // Matrix form: [x1  -R*x2] * [l1; l2] = t

                Eigen::Vector3f Rx1 = R * x1;

                Eigen::Matrix<float, 3, 2> A_tri;
                A_tri.col(0) = -Rx1;
                A_tri.col(1) = x2;

                // Solve overdetermined system A_tri * x = T using SVD or QR
                Eigen::Vector2f lambdas = A_tri.colPivHouseholderQr().solve(t);

                float lambda1 = lambdas[0];
                float lambda2 = lambdas[1];

                // Check if point is in front of both cameras and not infinitely far (parallax check)
                if (lambda1 > 0.0f && lambda2 > 0.0f && std::isfinite(lambda1))
                {
                    inFront++;
                    currentPoints.push_back(lambda1 * x1);
                }
            }

            if (inFront > maxInFront)
            {
                maxInFront = inFront;
                bestIdx = i;
                bestPoints = currentPoints;
            }
        }

        Eigen::Matrix4f pose = Eigen::Matrix4f::Identity();
        if (bestIdx >= 0)
        {
            pose.block<3, 3>(0, 0) = candidates[bestIdx].first;
            pose.block<3, 1>(0, 3) = candidates[bestIdx].second;

            std::cout << "Best solution: " << bestIdx << " with " << maxInFront
                      << "/" << tracks.size() << " points in front." << std::endl;

            std::cout << pose << std::endl;
        }
        else
        {
            std::cerr << "Failed to find valid pose." << std::endl;
        }

        return {pose, bestPoints};
    };

} // Namespace SfM::Solve