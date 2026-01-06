#include "../SfM.hpp"
#include <optional>

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
    Eigen::Matrix<T, 3, 3> calculateRotationMatrix(T rotX, T rotY, T rotZ)
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
    inline Mat4 blendCvMat()
    {
        // This flips Y (Up -> Down) and Z (Look -Z -> Look +Z)
        Mat4 blenderToCv = Mat4::Identity();
        blenderToCv(1, 1) = -1;
        blenderToCv(2, 2) = -1;
        return blenderToCv;
    }
} // Namespace SfM::util