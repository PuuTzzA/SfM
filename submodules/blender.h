#pragma once
#include <iostream>
#include <Eigen/Core>
#include "solve.h"

namespace SfM::Blender
{
        void printForBlender(const SfMResult &res);
}