#include "solve.hpp"
#include "../util/util.hpp"
#include <Eigen/SVD>
#include <iostream>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <thread>
#include <algorithm>

namespace SfM::solve
{
    class BundleAdjustmentConstraint
    {
    public:
        BundleAdjustmentConstraint(const Mat3 K, const Vec2 observation, const REAL weight) : m_K(K), m_observation{observation}, m_weight{weight} {};

        template <typename T>
        bool operator()(const T *const extrinsics, const T *const track, T *residuals) const
        {
            // extrinsics[0,1,2] are angle-axis rotation, magnitude = angle, direction = axis.
            // extrinsics[3,4,5] are translation.
            T p[3];
            ceres::AngleAxisRotatePoint(extrinsics, track, p); // TODO: implement oursevlves to be sure the error is not here
            p[0] += extrinsics[3];
            p[1] += extrinsics[4];
            p[2] += extrinsics[5];

            T x = p[0];
            T y = p[1];
            T z = p[2];

            p[0] = T(m_K(0, 0)) * x + T(m_K(0, 1)) * y + T(m_K(0, 2)) * z;
            p[1] = T(m_K(1, 0)) * x + T(m_K(1, 1)) * y + T(m_K(1, 2)) * z;
            p[2] = T(m_K(2, 0)) * x + T(m_K(2, 1)) * y + T(m_K(2, 2)) * z;

            if (ceres::abs(p[2]) < T(EPSILON))
            {
                std::cout << "returning false in ()" << std::endl;
                return false;
            }

            p[0] /= p[2];
            p[1] /= p[2];

            residuals[0] = (p[0] - T(m_observation[0])) * T(m_weight);
            residuals[1] = (p[1] - T(m_observation[1])) * T(m_weight);

            return true;
        }

        static ceres::CostFunction *create(const Mat3 &K, const Vec2 observation, const REAL weight)
        {
            return new ceres::AutoDiffCostFunction<BundleAdjustmentConstraint, 2, 6, 3>(new BundleAdjustmentConstraint(K, observation, weight));
        }

    private:
        const Mat3 m_K;
        const Vec2 m_observation;
        const REAL m_weight;
    };

    SfMResult bundleAdjustment(std::vector<Frame> &frames, Mat3 K, const int numTotTracks)
    {
        std::vector<Vec3> points3d;
        points3d.resize(numTotTracks, Vec3(0., 0., 1.));

        std::vector<REAL> extrinsics(frames.size() * 6, static_cast<REAL>(0)); // 3 * angle-axis, 3 * translation

        // Create Problem
        ceres::Problem problem;

        for (int i = 0, max = frames.size(); i < max; i++)
        {
            REAL *extrinsicPtr = &extrinsics[i * 6];

            for (const auto &point : frames[i].observations)
            {
                REAL *pointPtr = points3d[point.trackId].data();
                ceres::CostFunction *costFunction = BundleAdjustmentConstraint::create(K, point.point, 1.0);
                // problem.AddResidualBlock(costFunction, new ceres::HuberLoss(1.0), extrinsicPtr, pointPtr);
                problem.AddResidualBlock(costFunction, nullptr, extrinsicPtr, pointPtr);
            }
        }

        // Fix the first camera to remove Gauge Ambiguity (fixes the coordinate system origin)
        // If we don't do this, the whole world can drift freely.
        double *first_cam = &extrinsics[0];
        problem.SetParameterBlockConstant(first_cam);

        // Solve Problem
        ceres::Solver::Options options;
        options.linear_solver_type = ceres::DENSE_SCHUR;
        options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
        options.max_num_iterations = 256;
        options.num_threads = std::thread::hardware_concurrency();
        options.minimizer_progress_to_stdout = true;

        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);

        std::cout << "Running Bundle Adjustment on " << std::thread::hardware_concurrency() << " threads." << std::endl;
        std::cout << summary.FullReport() << "\n";

        // Extract Results
        Mat4 startTransform = SfM::util::calculateTransformationMatrixDeg(90, 0, 0, SfM::Vec3(0, 0, 0));

        SfMResult result;
        REAL scale = static_cast<REAL>(1);
        for (int i = 0; i < frames.size(); ++i)
        {
            REAL *params = &extrinsics[i * 6];

            REAL R_array[9];
            ceres::AngleAxisToRotationMatrix(params, R_array);

            Mat3 R;
            R << R_array[0], R_array[1], R_array[2],
                R_array[3], R_array[4], R_array[5],
                R_array[6], R_array[7], R_array[8];

            Vec3 t(params[3], params[4], params[5]);

            if (i == 1) // do this for i == 1 bc for i == 0 the translation is 0
            {
                REAL norm = t.norm();
                if (norm > EPSILON)
                {
                    scale = static_cast<REAL>(1.) / norm;
                }
            }

            t *= scale;

            t = R * -t;

            Mat4 viewMatrix = Mat4::Identity();
            viewMatrix.block<3, 3>(0, 0) = R;
            viewMatrix.block<3, 1>(0, 3) = t;

            result.extrinsics.push_back(startTransform * viewMatrix);
        }

        result.points = std::move(points3d);
        for (auto &p : result.points)
        {
            p = (util::blendCvMat() * startTransform * scale * p.homogeneous()).head<3>();
        }

        return result;
    };
} // Namespace SfM::solve