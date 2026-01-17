#include "detect.hpp"
#include <iostream>
#include "../io/file.hpp"
#include <omp.h>

namespace SfM::detect
{
    std::vector<Vec2> harrisCornerDetection(const Image<uchar> &image, const int blockSize, const int maxIter, const REAL maxDelta)
    {
        using H_REAL = float;

        size_t length = image.width * image.height;

        Image<H_REAL> gray;
        gray.data.resize(length);
        gray.width = image.width;
        gray.height = image.height;

#pragma omp parallel for simd
        for (int i = 0; i < length; i++)
        {
            int i2 = i * 3;
            H_REAL r = static_cast<H_REAL>(image.data[i2]) / static_cast<H_REAL>(255);
            H_REAL g = static_cast<H_REAL>(image.data[i2 + 1]) / static_cast<H_REAL>(255);
            H_REAL b = static_cast<H_REAL>(image.data[i2 + 2]) / static_cast<H_REAL>(255);

            gray.data[i] = 0.2125 * r + 0.7154 * g + 0.0721 * b; // Rec.709
            // gray.data[i] = 0.299 * r + 0.587 * g + 0.114 * b; // Rec.601
        }

        std::vector<H_REAL> xx(length);
        std::vector<H_REAL> yy(length);
        std::vector<H_REAL> xy(length);
// Calculate the harris respose
#pragma omp parallel for
        for (int y = 1; y < image.height - 1; y++)
        {
#pragma omp simd
            for (int x = 1; x < image.width - 1; x++)
            {
                int idx = y * image.width + x;

                // Calculate derivatives using sobel with kSize = 3.
                H_REAL dx = -gray.at(x - 1, y - 1) + gray.at(x + 1, y - 1)  //
                            - 2 * gray.at(x - 1, y) + 2 * gray.at(x + 1, y) //
                            - gray.at(x - 1, y + 1) + gray.at(x + 1, y + 1);

                H_REAL dy = -gray.at(x - 1, y - 1) - 2 * gray.at(x, y - 1) - gray.at(x + 1, y - 1) //
                            + gray.at(x - 1, y + 1) + 2 * gray.at(x, y + 1) + gray.at(x + 1, y + 1);

                xx[idx] = dx * dx;
                yy[idx] = dy * dy;
                xy[idx] = dx * dy;
            }
        }

        std::vector<H_REAL> xxS(length);
        std::vector<H_REAL> yyS(length);
        std::vector<H_REAL> xyS(length);
        // Blur the derivatives
        int lowerBound = -(blockSize - 1) / 2;         // inclusive
        int upperBound = lowerBound + (blockSize - 1); // inclusive
#pragma omp parallel for
        for (int y = -lowerBound; y < image.height - upperBound; y++)
        {
#pragma omp simd
            for (int x = -lowerBound; x < image.width - upperBound; x++)
            {
                int idx = y * image.width + x;

                xxS[idx] = 0;
                yyS[idx] = 0;
                xyS[idx] = 0;
                for (int v = lowerBound; v <= upperBound; v++)
                {
                    for (int u = lowerBound; u <= upperBound; u++)
                    {
                        int idx2 = (y + v) * image.width + (x + u);
                        xxS[idx] += xx[idx2];
                        yyS[idx] += yy[idx2];
                        xyS[idx] += xy[idx2];
                    }
                }
            }
        }

        std::vector<H_REAL> harris(length);
        H_REAL max = -std::numeric_limits<H_REAL>::max();
        // Calculate the harris response
#pragma omp parallel for reduction(max : max)
        for (int y = 1; y < image.height - 1; y++)
        {
            for (int x = 1; x < image.width - 1; x++)
            {
                int idx = y * image.width + x;

                H_REAL det = xxS[idx] * yyS[idx] - xyS[idx] * xyS[idx];
                H_REAL tr = xxS[idx] + yyS[idx];

                // calculate the Harris activation R=det(M)−k(trace(M))^2
                harris[idx] = det - 0.04 * (tr * tr);
                // harris[idx] *= 255;

                if (harris[idx] > max)
                {
                    max = harris[idx];
                }
            }
        }

        std::cout << "max: " << max << ", " << (max > -1.2e-10) << "\n";
// Threshhold the corners
#pragma omp parallel for
        for (int y = 1; y < image.height - 1; y++)
        {
#pragma omp simd
            for (int x = 1; x < image.width - 1; x++)
            {
                int idx = y * image.width + x;

                if (harris[idx] > max * 0.01)
                {
                    harris[idx] = 255;
                }
                else
                {
                    harris[idx] = gray.at(x, y) * 255;
                }
            }
        }

        gray.data = harris;
        auto cv = io::imageToCvMat(gray);
        // cv::dilate(cv, cv, cv::Mat());

        cv::imwrite("../../Data/______.png", cv);

        return {};
    }
}
