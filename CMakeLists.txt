cmake_minimum_required(VERSION 3.20)

project(SfM LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(SfM 
    main.cpp
    submodules/file.cpp
    # add more source files here
)

# If you need external libraries:
find_package(OpenCV REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

target_link_libraries(SfM
    PRIVATE
        ${OpenCV_LIBS}
        Eigen3::Eigen
)
