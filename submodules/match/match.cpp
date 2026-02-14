#include "match.hpp"
#include <iostream>
#include "../io/file.hpp"
#include "../util/util.hpp"
#include <omp.h>

namespace SfM::match
{
    std::vector<std::tuple<int, int>> match(const std::vector<Keypoint> &keypoints1, const std::vector<Keypoint> &keypoints2, const MATCHING_OPTIONS &options)
    {
        std::vector<std::tuple<int, int>> result;
        switch (options.algorithm)
        {
        case matchingAlgorithm::TWO_SIDED_MATHICHNG:
            result = matchTwoSided(keypoints1, keypoints2, options);
            break;
        default:
            throw std::invalid_argument("Match: Unknown matching algorithm. Was " + options.algorithm);
        }

        return result;
    };

    std::vector<std::tuple<int, int>> matchTwoSided(const std::vector<Keypoint> &keypoints1, const std::vector<Keypoint> &keypoints2, const MATCHING_OPTIONS &options)
    {
        int lenA = keypoints1.size();
        int lenB = keypoints2.size();

        Eigen::MatrixXf mat(lenA, lenB); // lenA rows, lenB columns, column-major
        Eigen::VectorXi matches1(lenA);
        Eigen::VectorXi matches2(lenB);

        // Calculate similarity matrix and find matches2
#pragma omp parallel for
        for (int idx2 = 0; idx2 < lenB; idx2++)
        {
            int maxIdx = -1;
            float maxSimilarity = 0;

            for (int idx1 = 0; idx1 < lenA; idx1++)
            {
                auto &kp1 = keypoints1[idx1];
                auto &kp2 = keypoints2[idx2];
                float similarity = options.similarityMetric(kp1.descriptor, kp2.descriptor);
                if (similarity > options.threshold && (kp1.point - kp2.point).squaredNorm() < options.maxDistancePxSquared)
                {
                    mat(idx1, idx2) = similarity;

                    if (similarity > maxSimilarity)
                    {
                        maxSimilarity = similarity;
                        maxIdx = idx1;
                    }
                }
                else
                {
                    mat(idx1, idx2) = 0;
                }
            }
            matches2[idx2] = maxIdx;
        }

        // Find matches1
#pragma omp parallel for
        for (int idx1 = 0; idx1 < lenA; idx1++)
        {
            int maxIdx = -1;
            float maxSimilarity = 0;

            for (int idx2 = 0; idx2 < lenB; idx2++)
            {
                if (mat(idx1, idx2) > maxSimilarity)
                {
                    maxSimilarity = mat(idx1, idx2);
                    maxIdx = idx2;
                }
            }
            matches1[idx1] = maxIdx;
        }

        std::vector<std::tuple<int, int>> matches;

        // Find matches where both agree
        for (int i = 0; i < lenA; i++)
        {
            int matchIn2 = matches1[i];
            if (matchIn2 != -1 && matches2[matchIn2] == i)
            {
                matches.push_back({i, matchIn2});
            }
        }

        return matches;
    }

    std::vector<std::tuple<int, int>> matchTwoSided2(const std::vector<Keypoint> &keypoints1, const std::vector<Keypoint> &keypoints2, const MATCHING_OPTIONS &options)
    {
        // About a factor or two slower than the other implementation
        int lenA = keypoints1.size();
        int lenB = keypoints2.size();

        // Removed the N*M matrix.
        // We only keep the best match indices for cross-checking.
        Eigen::VectorXi matches1(lenA);
        Eigen::VectorXi matches2(lenB);

        // -------------------------------------------------------
        // Pass 1: Find best match in A for every point in B
        // (Identical logic to the first loop of the original,
        // but we don't store the result in a matrix)
        // -------------------------------------------------------
#pragma omp parallel for
        for (int idx2 = 0; idx2 < lenB; idx2++)
        {
            int maxIdx = -1;
            float maxSimilarity = 0;

            for (int idx1 = 0; idx1 < lenA; idx1++)
            {
                auto &kp1 = keypoints1[idx1];
                auto &kp2 = keypoints2[idx2];
                float similarity = options.similarityMetric(kp1.descriptor, kp2.descriptor);

                // Exact same check as original
                if (similarity > options.threshold && (kp1.point - kp2.point).squaredNorm() < options.maxDistancePxSquared)
                {
                    if (similarity > maxSimilarity)
                    {
                        maxSimilarity = similarity;
                        maxIdx = idx1;
                    }
                }
            }
            matches2[idx2] = maxIdx;
        }

        // -------------------------------------------------------
        // Pass 2: Find best match in B for every point in A
        // (Instead of reading from a matrix, we RECOMPUTE values)
        // -------------------------------------------------------
#pragma omp parallel for
        for (int idx1 = 0; idx1 < lenA; idx1++)
        {
            int maxIdx = -1;
            float maxSimilarity = 0;

            for (int idx2 = 0; idx2 < lenB; idx2++)
            {
                // RECOMPUTATION START
                auto &kp1 = keypoints1[idx1];
                auto &kp2 = keypoints2[idx2];
                float similarity = options.similarityMetric(kp1.descriptor, kp2.descriptor);

                // In the original, mat(idx1, idx2) was set to 'similarity' ONLY if this check passed.
                // Otherwise, it was set to 0.
                if (similarity > options.threshold && (kp1.point - kp2.point).squaredNorm() < options.maxDistancePxSquared)
                {
                    // In original: if (mat(idx1, idx2) > maxSimilarity)
                    // Here: we know mat(idx1, idx2) would have been 'similarity'
                    if (similarity > maxSimilarity)
                    {
                        maxSimilarity = similarity;
                        maxIdx = idx2;
                    }
                }
                // Implicit else: in original, mat would be 0.
                // Since maxSimilarity starts at 0, 0 > maxSimilarity is false, so we do nothing.
            }
            matches1[idx1] = maxIdx;
        }

        // -------------------------------------------------------
        // Pass 3: Cross-Check (Mutual Nearest Neighbor)
        // (Identical to original)
        // -------------------------------------------------------
        std::vector<std::tuple<int, int>> matches;
        // Simple reserve to avoid reallocations, optional
        matches.reserve(std::min(lenA, lenB));

        for (int i = 0; i < lenA; i++)
        {
            int matchIn2 = matches1[i];
            // Check if A matched to something, and if that something matched back to A
            if (matchIn2 != -1 && matches2[matchIn2] == i)
            {
                matches.push_back({i, matchIn2});
            }
        }

        return matches;
    }
}
