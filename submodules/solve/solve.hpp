#pragma once
#include "../SfM.hpp"

namespace SfM::solve
{
    SfMResult eightPointAlgorithm(std::vector<std::vector<Vec2>> tracks);
} // Namespace SfM::solve