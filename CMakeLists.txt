cmake_minimum_required(VERSION 3.20)

project(SfM LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(SfM 
    main.cpp
    submodules/detect/detect.cpp
    submodules/detect/detectOpenCv.cpp
    submodules/io/file.cpp
    submodules/io/blender.cpp
    submodules/solve/bundleAdjustment.cpp
    submodules/solve/eightPoint.cpp
    submodules/solve/RANSAC.cpp
    submodules/solve/scene.cpp
    submodules/test/generate.cpp
    submodules/util/util.cpp
    # add more source files here
)

# If you need external libraries:
find_package(OpenMP REQUIRED)

find_package(OpenCV REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(Ceres REQUIRED)

find_path(TURBOJPEG_INCLUDE_DIR turbojpeg.h)
find_library(TURBOJPEG_LIBRARY NAMES turbojpeg)

if (TURBOJPEG_INCLUDE_DIR AND TURBOJPEG_LIBRARY)
    message(STATUS "Found TurboJPEG: ${TURBOJPEG_LIBRARY}")
else()
    message(FATAL_ERROR "Could not find TurboJPEG. Please install libjpeg-turbo8-dev")
endif()

target_include_directories(SfM
    PRIVATE
        ${TURBOJPEG_INCLUDE_DIR}
)

target_link_libraries(SfM
    PRIVATE
        OpenMP::OpenMP_CXX
        ${OpenCV_LIBS}
        Eigen3::Eigen
        Ceres::ceres
        ${TURBOJPEG_LIBRARY}  # Link the library found above
)