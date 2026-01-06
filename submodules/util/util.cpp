#include "util.hpp"
#include <iostream>
#include <math.h>

namespace SfM::util
{
    Mat4 calculateTransformationMatrixDeg(REAL rotX, REAL rotY, REAL rotZ, Vec3 translation)
    {
        rotX = rotX * M_PI / (REAL)180;
        rotY = rotY * M_PI / (REAL)180;
        rotZ = rotZ * M_PI / (REAL)180;

        return calculateTransformationMatrixRad(rotX, rotY, rotZ, translation);
    }

    Mat4 calculateTransformationMatrixRad(REAL rotX, REAL rotY, REAL rotZ, Vec3 translation)
    {
        Mat3 Rx;
        Rx << (REAL)1, (REAL)0, (REAL)0,
            (REAL)0, std::cos(rotX), -std::sin(rotX),
            (REAL)0, std::sin(rotX), std::cos(rotX);

        Mat3 Ry;
        Ry << std::cos(rotY), (REAL)0, std::sin(rotY),
            (REAL)0, (REAL)1, (REAL)0,
            -std::sin(rotY), (REAL)0, std::cos(rotY);

        Mat3 Rz;
        Rz << std::cos(rotZ), -std::sin(rotZ), (REAL)0,
            std::sin(rotZ), std::cos(rotZ), (REAL)0,
            (REAL)0, (REAL)0, (REAL)1;

        Mat3 R = Rz * Ry * Rx;
        Mat4 res = Mat4::Identity();
        res.block<3, 3>(0, 0) = R;
        res.block<3, 1>(0, 3) = translation;
        return res;
    }
} // Namespace SfM::util