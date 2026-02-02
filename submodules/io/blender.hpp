#pragma once
#include "../SfM.hpp"
#include "../scene.hpp"
#include <iostream>
#include <fstream>
#include <string>

namespace SfM::io
{
    /**
     * @brief Exports tracks to a file that can be imported into blender.
     * @note ALL INPUT EXTRINSICS AND POINTS ARE EXPECTED IN THE CV COORDINATE FRAME
     *
     * @param width Width of the images in px
     * @param height Height of the image in px
     * @param K Intrinsics matrix
     * @param cameraExtrinsics Vector of cameraExtrinsics
     * @param points Vector of 3d points
     * @param colors Vector of colors of the points in uchar [r, g, b]
     * @param path Output path
     * @param pathToImages Path to folder with images, RELATIVE TO PARAM path!
     */
    void exportTracksForBlender(int width,
                                int height,
                                Mat3 K,
                                std::vector<Mat4> &cameraExtrinsics,
                                std::vector<Vec3> &points,
                                std::vector<Vec3rgb> &colors,
                                std::string path,
                                std::string pathToImages = "");

    /**
     * @brief Exports tracks to a file that can be imported into blender.
     * @note ALL INPUT EXTRINSICS AND POINTS ARE EXPECTED IN THE CV COORDINATE FRAME#
     * @param scene Scene to export
     * @param path Output path
     * @param pathToImages Path to folder with images, RELATIVE TO PARAM path!
     */
    void exportSceneForBlender(Scene &scene, std::string path, std::string pathToImages = "");

    /**
     * @brief Exports tracks to a file that can be imported into blender.
     * @note ALL INPUT EXTRINSICS AND POINTS ARE EXPECTED IN THE CV COORDINATE FRAME
     *
     * @param cameraExtrinsics Vector of cameraExtrinsics
     * @param points Vector of 3d points
     * @param path Output path
     * @param pathToImages Path to folder with images, RELATIVE TO PARAM path!
     */
    void exportTracksForBlenderOld(std::vector<Mat4> &cameraExtrinsics, std::vector<Vec3> &points, std::string path, std::string pathToImages = "");
} // Namespace SfM::io