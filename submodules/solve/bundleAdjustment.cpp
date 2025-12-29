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
        BundleAdjustmentConstraint(const Mat3 &K, const Vec2 observation, const REAL weight) : m_K(K), m_observation{observation}, m_weight{weight} {};

        template <typename T>
        bool operator()(const T *const extrinsicsInverse, const T *const track, T *residuals) const
        {
            // extrinsicsInverse = [rx, ry, rz, tx, ty, tz]
            T p[3];
            ceres::AngleAxisRotatePoint(extrinsicsInverse, track, p); // TODO: implement oursevlves to be sure the error is not here
            p[0] += extrinsicsInverse[3];
            p[1] += extrinsicsInverse[4];
            p[2] += extrinsicsInverse[5];

            T x = p[0];
            T y = p[1];
            T z = p[2];

            p[0] = T(m_K(0, 0)) * x + T(m_K(0, 1)) * y + T(m_K(0, 2)) * z;
            p[1] = T(m_K(1, 0)) * x + T(m_K(1, 1)) * y + T(m_K(1, 2)) * z;
            p[2] = T(m_K(2, 0)) * x + T(m_K(2, 1)) * y + T(m_K(2, 2)) * z;

            if (ceres::abs(p[2]) < T(1e-10))
            {
                std::cout << "returning false in ()" << std::endl;
                return false;
            }

            p[0] /= p[2];
            p[1] /= p[2];

            residuals[0] = p[0] - T(m_observation[0]);
            residuals[1] = p[1] - T(m_observation[1]);

            return true;
        }

        static ceres::CostFunction *create(const Mat3 &K_inverse, const Vec2 observation, const REAL weight)
        {
            return new ceres::AutoDiffCostFunction<BundleAdjustmentConstraint, 2, 6, 3>(new BundleAdjustmentConstraint(K_inverse, observation, weight));
        }

    private:
        const Mat3 &m_K;
        const Vec2 m_observation;
        const REAL m_weight;
    };

    struct ExtrinsicInverse
    {
        REAL rx = static_cast<REAL>(0);
        REAL ry = static_cast<REAL>(0);
        REAL rz = static_cast<REAL>(0);
        REAL tx = static_cast<REAL>(0);
        REAL ty = static_cast<REAL>(0);
        REAL tz = static_cast<REAL>(0);
    };

    SfMResult bundleAdjustment(std::vector<Frame> &frames, Mat3 K, const int numTotKeypoints)
    {
        std::vector<Vec3> points3d;
        points3d.resize(numTotKeypoints, Vec3(7., 0., 1.));

        std::vector<ExtrinsicInverse> extrinsicsInverse;
        extrinsicsInverse.resize(frames.size());

        // Create Problem
        ceres::Problem problem;

        for (int i = 0, max = frames.size(); i < max; i++)
        {
            const auto &frame = frames[i];
            REAL *extrinsicPtr = &extrinsicsInverse[i].rx;

            for (const auto &point : frame.keypoints)
            {
                REAL *pointPtr = points3d[point.trackId].data();
                ceres::CostFunction *costFunction = BundleAdjustmentConstraint::create(K, point.point, 1.);
                problem.AddResidualBlock(costFunction, nullptr, extrinsicPtr, pointPtr);
            }
        }

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

        // Copy solutions
        SfMResult result;
        result.extrinsics.reserve(frames.size());
        result.points = std::move(points3d);

        std::for_each(result.points.begin(), result.points.end(), [](Vec3 &p)
                      { p = (util::blendCvMat() * p.homogeneous()).head<3>(); });

        for (int i = 0, max = frames.size(); i < max; i++)
        {
            ExtrinsicInverse eInv = extrinsicsInverse[i];
            Mat4 extrinsic = util::calculateTransformationMatrix(eInv.rx * 180. / M_PI, eInv.ry * 180. / M_PI, eInv.rz * 180. / M_PI, Vec3(eInv.tx, eInv.ty, eInv.tz));
            result.extrinsics.push_back(extrinsic);
        }

        return result;
    };
} // Namespace SfM::solve