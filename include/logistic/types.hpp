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

struct ProcessedData {
    std::vector<std::vector<double>> X;
    std::vector<int> y;
};
struct ModelConfig {
    double learning_rate = 0.01;
    int epochs = 1000;
    double l2_lambda = 0.0;
    int batch_size = 32;
};

} // namespace logistic
