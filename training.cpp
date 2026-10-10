#include "training.h"
#include <cmath>
#include <iostream>

// Sigmoid function
double sigmoid(double z) {
    // Prevent overflow
    if (z < -500) return 0.0;
    if (z > 500) return 1.0;
    return 1.0 / (1.0 + std::exp(-z));
}

// Main training function using Gradient Descent
std::vector<double> trainLogisticRegression(
    const std::vector<std::vector<double>>& X,
    const std::vector<int>& y,
    double learningRate,
    int epochs
) {
    int m = X.size();          // number of samples
    if (m == 0) return {};

    int n = X[0].size();       // number of features

    // Initialize weights to zero (weights[0] = bias)
    std::vector<double> weights(n + 1, 0.0);

    for (int epoch = 0; epoch < epochs; ++epoch) {
        std::vector<double> gradients(n + 1, 0.0);

        // Loop over all samples
        for (int i = 0; i < m; ++i) {
            // Calculate linear predictor z
            double z = weights[0]; // bias
            for (int j = 0; j < n; ++j) {
                z += weights[j + 1] * X[i][j];
            }

            double prediction = sigmoid(z);
            double error = prediction - y[i];

            // Accumulate gradients
            gradients[0] += error; // bias gradient
            for (int j = 0; j < n; ++j) {
                gradients[j + 1] += error * X[i][j];
            }
        }

        // Update weights
        for (int j = 0; j <= n; ++j) {
            weights[j] -= learningRate * (gradients[j] / m);
        }

        // Print progress every 100 epochs
        if ((epoch + 1) % 100 == 0 || epoch == 0) {
            double loss = 0.0;
            for (int i = 0; i < m; ++i) {
                double z = weights[0];
                for (int j = 0; j < n; ++j) {
                    z += weights[j + 1] * X[i][j];
                }
                double pred = sigmoid(z);
                // Binary cross-entropy loss
                loss += -y[i] * std::log(pred + 1e-15) 
                        - (1 - y[i]) * std::log(1.0 - pred + 1e-15);
            }
            loss /= m;
            std::cout << "Epoch " << (epoch + 1) 
                      << " | Loss: " << loss << std::endl;
        }
    }

    return weights;
}