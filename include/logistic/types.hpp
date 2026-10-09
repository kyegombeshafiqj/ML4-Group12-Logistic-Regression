#pragma once

#include <vector>

namespace logistic {

struct Sample {
    long id;
    int label;
    std::vector<double> features;
};

struct Dataset {
    std::vector<Sample> samples;
};

} // namespace logistic
