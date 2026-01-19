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
        Image<T> gray;
        gray.data.resize(image.width * image.height);
        gray.width = image.width;
        gray.height = image.height;

#pragma omp parallel for simd
        for (int i = 0; i < image.width * image.height; i++)
        {
            int i2 = i * 3;
            T r = static_cast<T>(image.data[i2]) / static_cast<T>(255);
            T g = static_cast<T>(image.data[i2 + 1]) / static_cast<T>(255);
            T b = static_cast<T>(image.data[i2 + 2]) / static_cast<T>(255);

            gray.data[i] = 0.2125 * r + 0.7154 * g + 0.0721 * b; // Rec.709
            // gray.data[i] = 0.299 * r + 0.587 * g + 0.114 * b; // Rec.601
        }

        return gray;
    }

    /**
     * @brief Element wise addition of the images. Not in place.
     */
    template <typename T>
    Image<T> add(const Image<T> &a, const Image<T> &b);

    //    /**
    //     * @brief Element wise subtraction of the images. Not in place.
    //     */
    //    template <typename T>
    //    inline Image<T> sub(const Image<T> &a, const Image<T> &b)
    //    {
    //        Image<T> out;
    //        out.width = a.width;
    //        out.height = a.height;
    //        out.data.resize(a.data.size());
    //
    // #pragma omp parallel for simd
    //        for (size_t i = 0; i < a.data.size(); ++i)
    //        {
    //            out.data[i] = a.data[i] - b.data[i];
    //        }
    //
    //        return out;
    //    }
    //
    //    /**
    //     * @brief Element wise subtraction of the images. Not in place.
    //     */
    //    template <typename T>
    //    inline Image<T> sub2(const Image<T> &a, const Image<T> &b)
    //    {
    //        Image<T> out;
    //        out.width = a.width;
    //        out.height = a.height;
    //        out.data.resize(a.data.size());
    //
    // #pragma omp parallel for
    //        for (int y = 0; y < a.height; y++)
    //        {
    //            int yOffset = y * a.width;
    // #pragma omp simd
    //            for (int x = 0; x < a.width; x++)
    //            {
    //                int idx = yOffset + x;
    //                out.data[idx] = a.data[idx] - b.data[idx];
    //            }
    //        }
    //        return out;
    //    }
    //
    //    /**
    //     * @brief Element wise subtraction of the images. Not in place.
    //     */
    //    template <typename T>
    //    inline Image<T> sub3(const Image<T> &a, const Image<T> &b)
    //    {
    //        Image<T> out;
    //        out.width = a.width;
    //        out.height = a.height;
    //        out.data.resize(a.data.size());
    //
    //        // TBB Parallel For
    //        tbb::parallel_for(tbb::blocked_range<size_t>(0, a.data.size()),
    //                          [&](const tbb::blocked_range<size_t> &r)
    //                          {
    //                              for (size_t i = r.begin(); i != r.end(); ++i)
    //                              {
    //                                  out.data[i] = a.data[i] - b.data[i];
    //                              }
    //                          });
    //
    //        return out;
    //    }
    //
    //    /**
    //     * @brief Element wise subtraction of the images. Not in place.
    //     */
    //    template <typename T>
    //    inline Image<T> sub4(const Image<T> &a, const Image<T> &b)
    //    {
    //        Image<T> out;
    //        out.width = a.width;
    //        out.height = a.height;
    //        out.data.resize(a.data.size());
    //
    //        // Parallelize over rows (y)
    //        tbb::parallel_for(tbb::blocked_range<int>(0, a.height),
    //                          [&](const tbb::blocked_range<int> &r)
    //                          {
    //                              for (int y = r.begin(); y != r.end(); ++y)
    //                              {
    //                                  int yOffset = y * a.width;
    //                                  for (int x = 0; x < a.width; x++)
    //                                  {
    //                                      int idx = yOffset + x;
    //                                      out.data[idx] = a.data[idx] - b.data[idx];
    //                                  }
    //                              }
    //                          });
    //
    //        return out;
    //    }
    //
    //    /**
    //     * @brief Element wise subtraction of the images. Not in place.
    //     */
    //    template <typename T>
    //    inline Image<T> sub5(const Image<T> &a, const Image<T> &b)
    //    {
    //        Image<T> out;
    //        out.width = a.width;
    //        out.height = a.height;
    //        out.data.resize(a.data.size());
    //
    //        // Parallelize over rows (y)
    //        tbb::parallel_for(tbb::blocked_range<int>(1, a.height - 1),
    //                          [&](const tbb::blocked_range<int> &r)
    //                          {
    //                              for (int y = r.begin(); y != r.end(); ++y)
    //                              {
    //                                  int yOffset = y * a.width;
    // #pragma omp simd
    //                                  for (int x = 0; x < a.width; x++)
    //                                  {
    //                                      int idx = yOffset + x;
    //                                      out.data[idx] = a.data[idx] - b.data[idx];
    //                                  }
    //                              }
    //                          });
    //
    //        return out;
    //    }

    /**
     * @brief Element wise subtraction.
     * @return T* Pointer to allocated memory. Caller must delete[].
     */
    template <typename T>
    inline T *sub(const Image<T> &a, const Image<T> &b)
    {
        size_t totalSize = a.data.size();
        T *outData = new T[totalSize];

#pragma omp parallel for simd
        for (size_t i = 0; i < totalSize; ++i)
        {
            outData[i] = a.data[i] - b.data[i];
        }

        return outData;
    }

    /**
     * @brief Element wise subtraction (2D Loop).
     * @return T* Pointer to allocated memory. Caller must delete[].
     */
    template <typename T>
    inline T *sub2(const Image<T> &a, const Image<T> &b)
    {
        size_t totalSize = a.width * a.height;
        T *outData = new T[totalSize];

#pragma omp parallel for
        for (int y = 0; y < a.height; y++)
        {
            int yOffset = y * a.width;
#pragma omp simd
            for (int x = 0; x < a.width; x++)
            {
                int idx = yOffset + x;
                outData[idx] = a.data[idx] - b.data[idx];
            }
        }
        return outData;
    }

    /**
     * @brief Element wise subtraction (TBB Linear).
     * @return T* Pointer to allocated memory. Caller must delete[].
     */
    template <typename T>
    inline T *sub3(const Image<T> &a, const Image<T> &b)
    {
        size_t totalSize = a.data.size();
        T *outData = new T[totalSize];

        // TBB Parallel For
        tbb::parallel_for(tbb::blocked_range<size_t>(0, totalSize),
                          [&](const tbb::blocked_range<size_t> &r)
                          {
                              for (size_t i = r.begin(); i != r.end(); ++i)
                              {
                                  outData[i] = a.data[i] - b.data[i];
                              }
                          });

        return outData;
    }

    /**
     * @brief Element wise subtraction (TBB 2D).
     * @return T* Pointer to allocated memory. Caller must delete[].
     */
    template <typename T>
    inline T *sub4(const Image<T> &a, const Image<T> &b)
    {
        size_t totalSize = a.width * a.height;
        T *outData = new T[totalSize];

        // Parallelize over rows (y)
        tbb::parallel_for(tbb::blocked_range<int>(0, a.height),
                          [&](const tbb::blocked_range<int> &r)
                          {
                              for (int y = r.begin(); y != r.end(); ++y)
                              {
                                  int yOffset = y * a.width;
                                  for (int x = 0; x < a.width; x++)
                                  {
                                      int idx = yOffset + x;
                                      outData[idx] = a.data[idx] - b.data[idx];
                                  }
                              }
                          });

        return outData;
    }

    /**
     * @brief Element wise subtraction (TBB + SIMD, skipping borders).
     * @return T* Pointer to allocated memory. Caller must delete[].
     * @note First and last rows will contain uninitialized memory (garbage).
     */
    template <typename T>
    inline T *sub5(const Image<T> &a, const Image<T> &b)
    {
        size_t totalSize = a.width * a.height;
        // Note: new T[] does not zero-initialize.
        // Since this loop skips y=0 and y=height-1, those rows will be garbage.
        T *outData = new T[totalSize];

        // Parallelize over rows (y), skipping first and last row
        tbb::parallel_for(tbb::blocked_range<int>(1, a.height - 1),
                          [&](const tbb::blocked_range<int> &r)
                          {
                              for (int y = r.begin(); y != r.end(); ++y)
                              {
                                  int yOffset = y * a.width;
#pragma omp simd
                                  for (int x = 0; x < a.width; x++)
                                  {
                                      int idx = yOffset + x;
                                      outData[idx] = a.data[idx] - b.data[idx];
                                  }
                              }
                          });

        return outData;
    }

    /**
     * @brief Element wise multiplication of the images. Not in place.
     */
    template <typename T>
    Image<T> mul(const Image<T> &a, const Image<T> &b);

    /**
     * @brief Element wise division of the images. Not in place.
     */
    template <typename T>
    Image<T> div(const Image<T> &a, const Image<T> &b);
} // Namespace SfM::util