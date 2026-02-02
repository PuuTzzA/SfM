#include "../SfM.hpp"
#include <optional>
#include <opencv2/opencv.hpp>
#include <omp.h>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>

namespace SfM::util
{
    /**
     * @brief Calculates the homeogeneous transformation matrix Rot_x @ Rot_y @ Rot_z + translation
     *
     * @param rotX Rotation around the x-axis in degrees
     * @param rotY Rotation around the y-axis in degrees
     * @param rotZ Rotation around the z-axis in degrees
     * @param translation translation
     */
    Mat4 calculateTransformationMatrixDeg(REAL rotX, REAL rotY, REAL rotZ, Vec3 translation);

    /**
     * @brief Calculates the homeogeneous transformation matrix Rot_x @ Rot_y @ Rot_z + translation
     *
     * @param rotX Rotation around the x-axis in radians
     * @param rotY Rotation around the y-axis in radians
     * @param rotZ Rotation around the z-axis in radians
     * @param translation translation
     */
    Mat4 calculateTransformationMatrixRad(REAL rotX, REAL rotY, REAL rotZ, Vec3 translation);

    /**
     * @brief Converts a struct Image to a cv::Mat.
     * @param image Input image
     * @return cv::Mat of the image (rgb if T == ucar, monochrome if T == REAL)
     */
    template <typename T>
    cv::Mat imageToCvMat(const Image<T> &image);

    /**
     * @brief Sample a point on an image with bilinear interpolation. Assumes img in BGR format
     */
    Vec3rgb getPixelBilinearUchar(const cv::Mat &img, Vec2 pos);

    /**
     * @brief Draws points on an image using UV coordinates and saves the result.
     *
     * @param image Input image
     * @param outputImagePath Path to the output image.
     * @param uvPoints Vector of points in pixel coordinates.
     * @param drawCircles If true, points are drawn as circles; otherwise as squares.
     * @param markerSize Radius/half-size of the point markers in pixels.
     * @param color Color of the drawn markers.
     * @return True if the image was successfully saved, false on failure.
     */
    bool drawPointsOnImage(const cv::Mat &image,
                           const std::string &outputImagePath,
                           const std::vector<Vec2> &uvPoints,
                           bool drawCircles = true,
                           int markerSize = 5,
                           cv::Scalar color = cv::Scalar(0, 0, 255));

    /**
     * @brief Visualizes tracked points over a range of frames.
     *
     * @param images Vector of input images.
     * @param tracks Vector of tracked points. Points in pixel coordinates.
     * @param startFrame use frames from startFrame ...
     * @param endFrame ... to endFrame
     * @param outputPath Path to output image.
     * @param markerSize Size of the markers.
     * @return True if the image was successfully saved, false on failure.
     */
    void drawCollageWithTracks(const std::vector<cv::Mat> &images,
                               const std::vector<std::vector<Vec2>> &tracks,
                               int startFrame,
                               int endFrame,
                               const std::string &outputPath,
                               int markerSize = 5);

    /**
     * @brief Calculates the homeogeneous transformation matrix Rot_x @ Rot_y @ Rot_z + translation
     *
     * @param rotX Rotation around the x-axis in radians
     * @param rotY Rotation around the y-axis in radians
     * @param rotZ Rotation around the z-axis in radians
     */
    template <typename T>
    Eigen::Matrix<T, 3, 3> calculateRotationMatrixRad(T rotX, T rotY, T rotZ)
    {
        using Mat3T = Eigen::Matrix<T, 3, 3>;
        using Mat4T = Eigen::Matrix<T, 4, 4>;

        Mat3T Rx;
        Rx << T(1), T(0), T(0),
            T(0), cos(rotX), -sin(rotX),
            T(0), sin(rotX), cos(rotX);

        // Rotation Y
        Mat3T Ry;
        Ry << cos(rotY), T(0), sin(rotY),
            T(0), T(1), T(0),
            -sin(rotY), T(0), cos(rotY);

        // Rotation Z
        Mat3T Rz;
        Rz << cos(rotZ), -sin(rotZ), T(0),
            sin(rotZ), cos(rotZ), T(0),
            T(0), T(0), T(1);

        return Rz * Ry * Rx;
    }

    /**
     * @brief Matrix that transforms from the blender coordinate frame (view: -Z, Y: up) to the standart CV coordinate frame (view: +Z, Y: down) and back
     */
    inline Mat4 blendCvMat4()
    {
        // This flips Y (Up -> Down) and Z (Look -Z -> Look +Z)
        Mat4 blenderToCv = Mat4::Identity();
        blenderToCv(1, 1) = -1;
        blenderToCv(2, 2) = -1;
        return blenderToCv;
    }

    /**
     * @brief Matrix that transforms from the blender coordinate frame (view: -Z, Y: up) to the standart CV coordinate frame (view: +Z, Y: down) and back
     */
    inline Mat3 blendCvMat3()
    {
        // This flips Y (Up -> Down) and Z (Look -Z -> Look +Z)
        Mat3 blenderToCv = Mat3::Identity();
        blenderToCv(1, 1) = -1;
        blenderToCv(2, 2) = -1;
        return blenderToCv;
    }

    /**
     * @brief Converts a camera pose from CV coordinates (view: -Z, Y: up) to the blender coordinate frame (view: +Z, Y: down) and back
     */
    inline Mat4 cvCameraToBlender(Mat4 cvCam)
    {
        return blendCvMat4() * cvCam * calculateTransformationMatrixDeg(180, 0, 0, Vec3(0, 0, 0));
    }

    /**
     * @brief Converts a rgb image to REAL.
     * @param image rgb image
     * @return image as REAL
     */
    template <typename T>
    inline Image<T> rgbToREAL(const Image<uchar> &image)
    {
        Image<T> gray(image.width, image.height);

        tbb::parallel_for(tbb::blocked_range<int>(0, image.height),
                          [&](const tbb::blocked_range<int> &r)
                          {
                              for (int y = r.begin(); y != r.end(); ++y)
                              {
                                  int yOffset = y * image.width;
#pragma omp simd
                                  for (int x = 0; x < image.width; x++)
                                  {
                                      int idx = yOffset + x;
                                      int idx2 = idx * 3;

                                      T r = static_cast<T>(image.data[idx2]) / static_cast<T>(255);
                                      T g = static_cast<T>(image.data[idx2 + 1]) / static_cast<T>(255);
                                      T b = static_cast<T>(image.data[idx2 + 2]) / static_cast<T>(255);

                                      gray.data[idx] = 0.2125 * r + 0.7154 * g + 0.0721 * b; // Rec.709
                                      // gray.data[idx] = 0.299 * r + 0.587 * g + 0.114 * b; // Rec.601
                                  }
                              }
                          });

        return gray;
    }

    /**
     * @brief Element wise addition. Not in place.
     */
    template <typename T>
    inline Image<T> add(const Image<T> &a, const Image<T> &b)
    {
        Image<T> outImg(a.width, a.height);

        tbb::parallel_for(tbb::blocked_range<int>(0, a.height),
                          [&](const tbb::blocked_range<int> &r)
                          {
                              for (int y = r.begin(); y != r.end(); ++y)
                              {
                                  int yOffset = y * a.width;
#pragma omp simd
                                  for (int x = 0; x < a.width; x++)
                                  {
                                      int idx = yOffset + x;
                                      outImg.data[idx] = a.data[idx] + b.data[idx];
                                  }
                              }
                          });

        return outImg;
    }

    /**
     * @brief Element wise subtraction. Not in place.
     */
    template <typename T>
    inline Image<T> sub(const Image<T> &a, const Image<T> &b)
    {
        Image<T> outImg(a.width, a.height);

        tbb::parallel_for(tbb::blocked_range<int>(0, a.height),
                          [&](const tbb::blocked_range<int> &r)
                          {
                              for (int y = r.begin(); y != r.end(); ++y)
                              {
                                  int yOffset = y * a.width;
#pragma omp simd
                                  for (int x = 0; x < a.width; x++)
                                  {
                                      int idx = yOffset + x;
                                      outImg.data[idx] = a.data[idx] - b.data[idx];
                                  }
                              }
                          });

        return outImg;
    }

    /**
     * @brief Element wise multiplication. Not in place.
     */
    template <typename T>
    inline Image<T> mul(const Image<T> &a, const Image<T> &b)
    {
        Image<T> outImg(a.width, a.height);

        tbb::parallel_for(tbb::blocked_range<int>(0, a.height),
                          [&](const tbb::blocked_range<int> &r)
                          {
                              for (int y = r.begin(); y != r.end(); ++y)
                              {
                                  int yOffset = y * a.width;
#pragma omp simd
                                  for (int x = 0; x < a.width; x++)
                                  {
                                      int idx = yOffset + x;
                                      outImg.data[idx] = a.data[idx] * b.data[idx];
                                  }
                              }
                          });

        return outImg;
    }

    /**
     * @brief Element wise division. Not in place.
     */
    template <typename T>
    inline Image<T> div(const Image<T> &a, const Image<T> &b)
    {
        Image<T> outImg(a.width, a.height);

        tbb::parallel_for(tbb::blocked_range<int>(0, a.height),
                          [&](const tbb::blocked_range<int> &r)
                          {
                              for (int y = r.begin(); y != r.end(); ++y)
                              {
                                  int yOffset = y * a.width;
#pragma omp simd
                                  for (int x = 0; x < a.width; x++)
                                  {
                                      int idx = yOffset + x;
                                      outImg.data[idx] = a.data[idx] / b.data[idx];
                                  }
                              }
                          });

        return outImg;
    }

    /**
     * @brief Element wise multiplication. Not in place.
     */
    template <typename T>
    inline Image<T> mulScalar(const Image<T> &a, T scalar)
    {
        Image<T> outImg(a.width, a.height);

        tbb::parallel_for(tbb::blocked_range<int>(0, a.height),
                          [&](const tbb::blocked_range<int> &r)
                          {
                              for (int y = r.begin(); y != r.end(); ++y)
                              {
                                  int yOffset = y * a.width;
#pragma omp simd
                                  for (int x = 0; x < a.width; x++)
                                  {
                                      int idx = yOffset + x;
                                      outImg.data[idx] = a.data[idx] * scalar;
                                  }
                              }
                          });

        return outImg;
    }

    /**
     * @brief Returns a vector containing the 1D kernel weights
     */
    template <typename T>
    inline std::vector<T> createGaussianKernel(REAL sigma, int order)
    {
        int radius = static_cast<int>(4.0 * sigma + 0.5);
        int size = 2 * radius + 1;

        std::vector<T> kernel;
        kernel.reserve(size);
        T sum = 0;
        T sigmaSqr = sigma * sigma;
        T twoSigmaSqr = 2 * sigmaSqr;

        for (int x = -radius; x <= radius; x++)
        {
            REAL g = std::exp(-(x * x) / twoSigmaSqr);

            if (order == 0) // Zeroth derivative = Gaussian function
            {
                kernel.push_back(g);
                sum += g;
            }
            else if (order == 1) // First derivative
            {
                kernel.push_back(-(x / sigmaSqr) * g);
            }
            // Higher derivatives go here
        }

        if (order == 0 && std::abs(sum) > EPSILON)
        {
            // Normalize values. Only for order = 0
            for (int i = 0; i < size; i++)
            {
                kernel[i] /= sum;
            }
        }

        return kernel;
    }

    /**
     * @brief Computes the gaussian blur of an Image<T>
     * @param image Image to be blurred
     * @param sigma (sigmaX, sigmaY) sigmas for the axis
     * @param order (orderX, orderY) orders of the derivative used in the convolution (0 = normal gaussian blur). Only 0 or 1 supported for now.
     * @return Returns blurred Image<T>
     */
    template <typename T>
    inline Image<T> gaussianBlur(const Image<T> &image, Vec2 sigma, Vec2I order = Vec2I(0, 0))
    {
        if ((order[0] != 0 && order[0] != 1) || (order[1] != 0 && order[1] != 1))
        {
            std::cerr << "GAUSSIAN BLUR: Order has to be 0 or 1, was: " << order[0] << ", " << order[1] << std::endl;
            return Image<T>(0, 0);
        }

        auto kernelX = createGaussianKernel<T>(sigma[0], order[0]);
        int radiusX = (kernelX.size() - 1) / 2;
        auto kernelY = createGaussianKernel<T>(sigma[1], order[1]);
        int radiusY = (kernelY.size() - 1) / 2;

        T *temp = new T[image.width * image.height];
        Image<T> outImg(image.width, image.height);

// Horizontal Convolution
#pragma omp parallel for
        for (int y = 0; y < image.height; y++)
        {
            int yOffset = y * image.width;
#pragma omp parallel for simd
            for (int x = 0; x < image.width; x++)
            {
                T val = 0;

                for (int r = -radiusX; r <= radiusX; r++)
                {
                    int xOffset = std::clamp(x + r, 0, image.width - 1);
                    val += kernelX[r + radiusX] * image.data[yOffset + xOffset];
                }

                temp[yOffset + x] = val;
            }
        }

// Vertical Convolution
#pragma omp parallel for
        for (int x = 0; x < image.width; x++)
        {
#pragma omp parallel for simd
            for (int y = 0; y < image.height; y++)
            {
                T val = 0;

                for (int r = -radiusY; r <= radiusY; r++)
                {
                    int yOffset = std::clamp(y + r, 0, image.height - 1);
                    val += kernelY[r + radiusY] * temp[yOffset * image.width + x];
                }

                outImg.data[y * image.width + x] = val;
            }
        }

        delete[] temp;
        return outImg;
    }

    /**
     * @brief Computes the gaussian blur of an Image<uchar>
     * @param image Image to be blurred
     * @param sigma (sigmaX, sigmaY) sigmas for the axis
     * @param order (orderX, orderY) orders of the derivative used in the convolution (0 = normal gaussian blur). Only 0 or 1 supported for now.
     * @return Returns blurred Image<T>
     */
    template <>
    inline Image<uchar> gaussianBlur(const Image<uchar> &image, Vec2 sigma, Vec2I order)
    {
        if (order[0] != 0 || order[1] != 0)
        {
            std::cerr << "GAUSSIAN BLUR: Order has to be 0 for Image<uchar>! Was: " << order[0] << ", " << order[1] << std::endl;
            std::cerr << "GAUSSIAN BLUR: Use REAL for derivatives." << std::endl;
            return Image<uchar>(0, 0);
        }

        auto kernelX = createGaussianKernel<REAL>(sigma[0], order[0]);
        int radiusX = (kernelX.size() - 1) / 2;
        auto kernelY = createGaussianKernel<REAL>(sigma[1], order[1]);
        int radiusY = (kernelY.size() - 1) / 2;

        REAL *temp = new REAL[image.width * image.height * 3];
        Image<uchar> outImg(image.width, image.height, 3);

// Horizontal Convolution
#pragma omp parallel for
        for (int y = 0; y < image.height; y++)
        {
            int yOffset = y * image.width;
#pragma omp parallel for simd
            for (int x = 0; x < image.width; x++)
            {
                REAL valR = 0;
                REAL valG = 0;
                REAL valB = 0;

                for (int r = -radiusX; r <= radiusX; r++)
                {
                    int xOffset = std::clamp(x + r, 0, image.width - 1);
                    valR += kernelX[r + radiusX] * static_cast<REAL>(image.data[3 * (yOffset + xOffset)]);
                    valG += kernelX[r + radiusX] * static_cast<REAL>(image.data[3 * (yOffset + xOffset) + 1]);
                    valB += kernelX[r + radiusX] * static_cast<REAL>(image.data[3 * (yOffset + xOffset) + 2]);
                }

                temp[3 * (yOffset + x)] = valR;
                temp[3 * (yOffset + x) + 1] = valG;
                temp[3 * (yOffset + x) + 2] = valB;
            }
        }

// Vertical Convolution
#pragma omp parallel for
        for (int x = 0; x < image.width; x++)
        {
#pragma omp parallel for simd
            for (int y = 0; y < image.height; y++)
            {
                REAL valR = 0;
                REAL valG = 0;
                REAL valB = 0;

                for (int r = -radiusY; r <= radiusY; r++)
                {
                    int yOffset = std::clamp(y + r, 0, image.height - 1);
                    valR += kernelY[r + radiusY] * temp[3 * (yOffset * image.width + x)];
                    valG += kernelY[r + radiusY] * temp[3 * (yOffset * image.width + x) + 1];
                    valB += kernelY[r + radiusY] * temp[3 * (yOffset * image.width + x) + 2];
                }

                outImg.data[3 * (y * image.width + x)] = static_cast<uchar>(valR);
                outImg.data[3 * (y * image.width + x) + 1] = static_cast<uchar>(valG);
                outImg.data[3 * (y * image.width + x) + 2] = static_cast<uchar>(valB);
            }
        }

        delete[] temp;
        return outImg;
    }
} // Namespace SfM::util