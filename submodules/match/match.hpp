#pragma once
#include "../SfM.hpp"
#include <functional>
#include <opencv2/opencv.hpp>

namespace SfM::match
{
    /**
     * @brief Creates a new frame from a list of Keypoints
     * @param keypoints Vector of Keypoints
     * @return A newly initialized frame
     */
    Frame createFirstFrameFromKeypoints(std::vector<Keypoint> &keypoints);

    using similarityFunction = std::function<float(const std::vector<float> &, const std::vector<float> &)>;

    /**
     * @brief Computes the dot product of two vectors
     */
    inline float dotProduct(const std::vector<float> &a, const std::vector<float> &b)
    {
        float result = 0.0f;

        for (size_t i = 0; i < a.size(); ++i)
        {
            result += a[i] * b[i];
        }
        return result;
    }

    /**
     * @brief Computs the cosine similarity of two vectors
     */
    inline float cosineSimilarity(const std::vector<float> &a, const std::vector<float> &b)
    {
        float dot = 0.0f;
        float normA = 0.0f;
        float normB = 0.0f;

        for (size_t i = 0; i < a.size(); ++i)
        {
            dot += a[i] * b[i];
            normA += a[i] * a[i];
            normB += b[i] * b[i];
        }

        return dot / (std::sqrt(normA) * std::sqrt(normB));
    }

    /**
     * @brief Finds matching keypoints between frames
     * The matching is performed on a two-sided
     * @param keypoints1 Vector of keypoints to match
     * @param keypoints2 Vector of keypoints to match
     * @param threshhold Min similarity score 
     * @param similarityFunction Function that computes a similarity measure between two vectors (default cosine similarity)
     * @return Vector of matches (<int, int> = index in first, index in second)
     */
    std::vector<std::tuple<int, int>> matchTwoSided(std::vector<Keypoint> &keypoints1, std::vector<Keypoint> &keypoints2, float threshhold = 0.5, similarityFunction similarityFunction = cosineSimilarity);

    // vl hungarian method

} // Namespace SfM::match