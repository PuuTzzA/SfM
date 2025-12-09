#include <iostream>
#include "submodules/file.h"
#include <opencv2/core.hpp>

int main()
{
    std::cout << "Hello SfM" << std::endl;
    SfM::loadFile("Hello from inside the Submodule");
    std::cout << "OpenCV version: " << CV_VERSION << std::endl;
    return 0;
}
