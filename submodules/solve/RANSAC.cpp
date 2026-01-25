#include "solve.hpp"
#include "../util/util.hpp"
#include <iostream>
#include <chrono>
#include <algorithm>
#include <random>
#include <cmath>
#include <atomic>
#include <mutex>
#include <tbb/tbb.h>

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

    std::vector<int> RANSAC_OLD(const std::vector<Vec2> &x, const std::vector<Vec2> &y, const Mat3 &K, const RANSAC_OPTIONS &options)
    {
        if (x.size() != y.size() || x.size() < options.minN)
        {
            std::cerr << "RANSAC: x and y have to have the same size >= options.minN. x.size() was " << x.size() << std::endl;
            return {};
        }

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

    std::vector<int> RANSAC(const std::vector<Vec2> &x, const std::vector<Vec2> &y, const Mat3 &K, const RANSAC_OPTIONS &options, const bool verbose)
    {
        if (x.size() != y.size() || x.size() < options.minN)
            return {};

        auto start = std::chrono::steady_clock::now();
        auto limit = std::chrono::milliseconds(options.maxTimeMs);

        // Shared Output
        std::vector<int> bestInliers;
        EightPointResult bestResult;
        REAL bestError = std::numeric_limits<REAL>::infinity();
        tbb::spin_mutex myMutex; // TBB spin mutex is faster for short critical sections

        std::atomic<int> maxIter(options.maxIter);
        std::atomic<int> iterationsDone(0);

        // TBB Parallel For
        // We iterate from 0 to options.maxIter (initial guess).
        // We will cancel effectively by checking atomic flags.
        tbb::parallel_for(tbb::blocked_range<int>(0, options.maxIter),
                          [&](const tbb::blocked_range<int> &r)
                          {
                              // Thread-Local Setup
                              std::default_random_engine rng(std::chrono::system_clock::now().time_since_epoch().count() + tbb::this_task_arena::current_thread_index());
                              std::vector<int> indices(x.size());
                              std::iota(indices.begin(), indices.end(), 0);

                              std::vector<Vec2> currentX(options.minN);
                              std::vector<Vec2> currentY(options.minN);
                              std::vector<int> currentInliers;
                              currentInliers.reserve(x.size());

                              for (int i = r.begin(); i != r.end(); ++i)
                              {
                                  // 1. Early Exit Checks
                                  if (iterationsDone.load() >= maxIter.load())
                                  {
                                      // If we passed the adaptive limit, stop processing this block
                                      break;
                                  }

                                  // Time check (occasional)
                                  if (i % 50 == 0)
                                  {
                                      if (std::chrono::steady_clock::now() - start > limit)
                                      {
                                          maxIter.store(0);
                                          break;
                                      }
                                  }

                                  iterationsDone++;

                                  // 2. Shuffle & Model
                                  std::shuffle(indices.begin(), indices.end(), rng);
                                  for (int k = 0; k < options.minN; k++)
                                  {
                                      currentX[k] = x[indices[k]];
                                      currentY[k] = y[indices[k]];
                                  }

                                  auto currentResult = options.model(currentX, currentY, x, y);

                                  // 3. Evaluate Inliers
                                  currentInliers.clear();
                                  REAL currentTotError = 0;
                                  for (size_t k = 0; k < x.size(); k++)
                                  {
                                      auto error = options.loss(K, x[k], y[k], currentResult.points[k], currentResult.pose);
                                      if (error < options.maxSquaredError)
                                      {
                                          currentInliers.push_back(k);
                                          currentTotError += error;
                                      }
                                  }

                                  // 4. Update Best (Critical Section)
                                  // Quick check before locking
                                  bool possibleUpdate = false;
                                  {
                                      // Unsafe read is okay for optimization here
                                      // We just want to avoid locking if the result is garbage
                                      if (currentInliers.size() > 0)
                                          possibleUpdate = true;
                                  }

                                  if (possibleUpdate)
                                  {
                                      tbb::spin_mutex::scoped_lock lock(myMutex);

                                      if (currentInliers.size() > bestInliers.size() ||
                                          (currentInliers.size() == bestInliers.size() && currentTotError < bestError))
                                      {
                                          bestError = currentTotError;
                                          bestInliers = currentInliers; // Copy
                                          bestResult = currentResult;

                                          // Recalculate Max Iter
                                          REAL inlierRatio = static_cast<REAL>(bestInliers.size()) / static_cast<REAL>(x.size());
                                          REAL succesProb = std::pow(inlierRatio, options.minN);

                                          if (succesProb >= EPSILON)
                                          {
                                              int newMax = safeCastToInt(std::ceil(std::log(1. - options.successProb) / std::log(1 - succesProb)));
                                              if (verbose)
                                              {
                                                  std::cout << "RANSAC: found better model with " << bestInliers.size() << " inliers that have a total error of " << currentTotError << ", maxIter adjusted to " << std::min(newMax, maxIter.load()) << ".\n";
                                              }
                                              if (newMax < maxIter.load())
                                              {
                                                  maxIter.store(newMax);
                                              }
                                          }
                                      }
                                  }
                              }
                          });

        if (verbose)
        {
            std::cout << "RANSAC: finished after " << iterationsDone << " iterations (" << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count() << "ms)." << std::endl;
        }
        return bestInliers;
    }
} // namespace SfM::solve