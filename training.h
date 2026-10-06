#ifndef TRAINING_H
#define TRAINING_H

#include <vector>

// Train logistic regression using Gradient Descent
// Returns the learned weights (bias is weights[0])
std::vector<double> trainLogisticRegression(
    const std::vector<std::vector<double>>& X,
    const std::vector<int>& y,
    double learningRate = 0.1,
    int epochs = 1000
);

// Helper: sigmoid function
double sigmoid(double z);

#endif