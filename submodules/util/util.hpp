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
    Mat4 calculateTransformationMatrix(REAL rotX, REAL rotY, REAL rotZ, Vec3 translation);

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