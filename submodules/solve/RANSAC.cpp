#include "solve.hpp"
#include "../util/util.hpp"
#include <iostream>
#include <chrono>
#include <algorithm>
#include <random>
#include <cmath>

namespace SfM::solve
{
    int safeCastToInt(REAL value)
    {
        if (std::isnan(value))
        {
            return 0;
        }

        if (value >= static_cast<REAL>(std::numeric_limits<int>::max()))
        {
            return std::numeric_limits<int>::max();
        }

        if (value <= static_cast<double>(std::numeric_limits<int>::min()))
        {
            return std::numeric_limits<int>::min();
        }

        return static_cast<int>(value);
    }

    std::vector<int> RANSAC(const std::vector<Vec2> &x, const std::vector<Vec2> &y, const Mat3 &K, const RANSAC_OPTIONS &options)
    {
        auto limit = std::chrono::milliseconds(options.maxTimeMs);
        auto start = std::chrono::steady_clock::now();

        std::vector<int> bestInliers;
        EightPointResult bestResult;
        REAL bestError = std::numeric_limits<REAL>::infinity();

        auto rng = std::default_random_engine{};
        std::vector<int> indices(x.size());
        for (int i = 0; i < x.size(); i++)
        {
            indices[i] = i;
        }
        std::vector<Vec2> currentX(options.minN);
        std::vector<Vec2> currentY(options.minN);
        EightPointResult currentResult;
        std::vector<int> currentInliers;

        int iterations = 0;
        int maxIter = options.maxIter;
        while (iterations < maxIter)
        {
            // early exit due to runtime or maximum iterations
            {
                auto now = std::chrono::steady_clock::now();
                if (now - start > limit)
                {
                    std::cout << "RANSAC: maximum time (" << options.maxTimeMs << "ms) reached after " << iterations - 1 << " iterations." << std::endl;
                    return bestInliers;
                }

                if (iterations > options.maxIter)
                {
                    std::cout << "RANSAC: maximum iterations (" << options.maxIter << ") reached after " << iterations - 1 << " iterations." << std::endl;
                    return bestInliers;
                }
            }

            std::shuffle(indices.begin(), indices.end(), rng);
            for (int i = 0; i < options.minN; i++)
            {
                currentX[i] = x[indices[i]];
                currentY[i] = y[indices[i]];
            }

            currentResult = options.model(currentX, currentY, x, y);

            currentInliers.clear();
            REAL currentTotError = 0;
            for (int i = 0; i < x.size(); i++)
            {
                auto error = options.loss(K, x[i], y[i], currentResult.points[i], currentResult.pose);
                if (error < options.maxSquaredError)
                {
                    currentInliers.push_back(i);
                    currentTotError += error;
                }
            }

            if (currentInliers.size() > bestInliers.size() || (currentInliers.size() > 0 && currentInliers.size() == bestInliers.size() && currentTotError < bestError))
            {
                bestError = currentTotError;
                bestInliers = std::move(currentInliers);
                bestResult = std::move(currentResult);
                REAL inlierRatio = static_cast<REAL>(bestInliers.size()) / static_cast<REAL>(x.size());
                REAL succesProb = std::pow(inlierRatio, options.minN);

                if (succesProb < EPSILON) // Do this because otherwise log(1) = 0 => division = -inf and maxIter = -2147483648
                {
                    maxIter = options.maxIter;
                }
                else
                {
                    maxIter = std::min(maxIter, safeCastToInt(std::ceil(std::log(1. - options.successProb) / std::log(1 - succesProb))));
                }
                std::cout << "RANSAC: found better model with " << bestInliers.size() << " inliers that have a total error of " << currentTotError << ", maxIter adjusted to " << maxIter << ".\n";
            }

            iterations++;

            // wikipedia describes the algorithm a bit differently, but we want to find the biggest set of inliers and not the set with the smallest error
            /* if (currentInliers.size() > m_minInliers)
            {
                gatherPointsFromIndices(x, currentX, y, currentY, currentInliers, currentInliers.size());
                currentResult = eightPointAlgorithm(currentX, currentY);

                REAL currentError = 0;
                for (int i = 0; i < currentInliers.size(); i++)
                {
                    currentError += m_loss(K, currentX[i], currentResult.points[i], currentResult.pose);
                }
                if (currentError < bestError)
                {
                    bestInliers = std::move(currentInliers);
                    bestResult = std::move(currentResult);
                    bestError = currentError;
                }
            } */
        }

        std::cout << "RANSAC: finished after " << iterations << " iterations." << std::endl;
        return bestInliers;
    }

} // namespace SfM::solve