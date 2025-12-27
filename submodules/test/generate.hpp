#include "../SfM.hpp"
#include <optional>

namespace SfM::test
{
    /**
     * @brief Generates random points and returns the corresponding Tracks.
     * NOTE: The locaiotns and cameraExtrinsics are given in Blender Coordinates (look: -Z, Y: up)
     *
     * @param cameraExtrinsics Vector of camera extrinsics
     * @param cameraIntrinsics Matrix of camera intrinsics
     * @param pointsLocation Average position around which points are distributed
     * @param pointsRadius Maximum absolute distance in x, y, z of the points to the pointsLocation
     * @param points out 3D location of the generated points, as a ground truth value
     * @param numPoints Number of generated points
     * @param detectionError Random Jiggle that is added to the observations (simulates non-perfect camera/keypoint detection)
     * @return Vector of Tracks from the specified camera poses
     */
    std::vector<Track> generateRandomPointsTracks(std::vector<Mat4> &cameraExtrinsics,
                                                  Mat3 cameraIntrinsics,
                                                  Vec3 pointsLocation,
                                                  Vec3 pointsRadius,
                                                  std::optional<std::reference_wrapper<std::vector<Vec3>>> points = std::nullopt,
                                                  int numPoints = 20,
                                                  Vec2 detectionError = Vec2(0, 0));

    /**
     * @brief Generates random points and returns the corresponding Tracks.
     * NOTE: The locaiotns and cameraExtrinsics are given in Blender Coordinates (look: -Z, Y: up)
     *
     * @param cameraExtrinsics Vector of camera extrinsics
     * @param cameraIntrinsics Matrix of camera intrinsics
     * @param pointsLocation Average position around which points are distributed
     * @param pointsRadius Maximum absolute distance in x, y, z of the points to the pointsLocation
     * @param points out 3D location of the generated points, as a ground truth value
     * @param numPoints Number of generated points
     * @param detectionError Random Jiggle that is added to the observations (simulates non-perfect camera/keypoint detection)
     * @return Vector of Frames from the specified camera poses
     */
    std::vector<Frame> generateRandomPointsFrames(std::vector<Mat4> &cameraExtrinsics,
                                                  Mat3 cameraIntrinsics,
                                                  Vec3 pointsLocation,
                                                  Vec3 pointsRadius,
                                                  std::optional<std::reference_wrapper<std::vector<Vec3>>> points = std::nullopt,
                                                  int numPoints = 20,
                                                  Vec2 detectionError = Vec2(0, 0));

} // Namespace SfM::test