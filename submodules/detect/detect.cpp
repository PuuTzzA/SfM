#include "detect.hpp"
#include <iostream>
#include "../io/file.hpp"
#include "../util/util.hpp"
#include <omp.h>

namespace SfM::detect
{
    std::vector<Vec2> harrisCornerDetection(const Image<uchar> &image, const int blockSize, const int maxIter, const REAL maxDelta)
    {
        using H_REAL = float;

        size_t length = image.width * image.height;

        Image<H_REAL> gray = util::rgbToREAL<H_REAL>(image);

        // Calculate derivatives using sobel with kSize = 3.
        H_REAL *xx = new H_REAL[length];
        H_REAL *yy = new H_REAL[length];
        H_REAL *xy = new H_REAL[length];
#pragma omp parallel for
        for (int y = 1; y < image.height - 1; y++)
        {
            int yOffset = y * image.width;
#pragma omp simd
            for (int x = 1; x < image.width - 1; x++)
            {
                int idx = yOffset + x;

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

        // Blur the derivatives and calculate the harris response
        Image<H_REAL> harris(image.width, image.height);
        H_REAL max = -std::numeric_limits<H_REAL>::max();
        int lowerBound = -(blockSize - 1) / 2;         // inclusive
        int upperBound = lowerBound + (blockSize - 1); // inclusive
#pragma omp parallel for reduction(max : max)
        for (int y = -lowerBound; y < image.height - upperBound; y++)
        {
            int yOffset = y * image.width;
#pragma omp simd reduction(max : max)
            for (int x = -lowerBound; x < image.width - upperBound; x++)
            {
                int idx = yOffset + x;

                H_REAL sumXx = 0;
                H_REAL sumYy = 0;
                H_REAL sumXy = 0;
                for (int v = lowerBound; v <= upperBound; v++)
                {
                    for (int u = lowerBound; u <= upperBound; u++)
                    {
                        int idx2 = (y + v) * image.width + (x + u);
                        sumXx += xx[idx2];
                        sumYy += yy[idx2];
                        sumXy += xy[idx2];
                    }
                }

                H_REAL det = sumXx * sumYy - sumXy * sumXy;
                H_REAL tr = sumXx + sumYy;
                // calculate the Harris activation R=det(M)−k(trace(M))^2
                harris.data[idx] = det - 0.04 * (tr * tr);
                // harris[idx] *= 255;

                if (harris.data[idx] > max)
                {
                    max = harris.data[idx];
                }
            }
        }

        // std::cout << "max: " << max << ", " << (max > -1.2e-10) << "\n";
// Threshhold the corners
#pragma omp parallel for
        for (int y = 1; y < image.height - 1; y++)
        {
#pragma omp simd
            for (int x = 1; x < image.width - 1; x++)
            {
                int idx = y * image.width + x;

                if (harris.data[idx] > max * 0.01)
                {
                    harris.data[idx] = 255;
                }
                else
                {
                    harris.data[idx] = gray.at(x, y) * 255;
                }
            }
        }

        // auto cv = io::imageToCvMat(harris);
        // cv::dilate(cv, cv, cv::Mat());
        // cv::imwrite("../../Data/______.png", cv);

        delete[] xx;
        delete[] yy;
        delete[] xy;
        return {};
    }

    void SIFT(const Image<uchar> &image)
    {
        using S_REAL = float;

        size_t length = image.width * image.height;

        Image<S_REAL> gray = util::rgbToREAL<S_REAL>(image);
    }
}
