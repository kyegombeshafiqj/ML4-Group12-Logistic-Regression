#include <iostream>
#include <vector>
#include "training.h"

int main() {
    // Small fake dataset for testing (2 features)
    // Class 0 points
    std::vector<std::vector<double>> X = {
        {1.0, 2.0},
        {1.5, 1.8},
        {2.0, 2.2},
        {5.0, 6.0},
        {5.5, 5.8},
        {6.0, 6.2}
    };

    std::vector<int> y = {0, 0, 0, 1, 1, 1};

    std::cout << "Starting training...\n" << std::endl;

    // Train the model
    std::vector<double> weights = trainLogisticRegression(X, y, 0.5, 500);

    std::cout << "\nTraining finished!\n" << std::endl;
    std::cout << "Learned weights:" << std::endl;
    std::cout << "Bias (w0): " << weights[0] << std::endl;
    for (size_t i = 1; i < weights.size(); ++i) {
        std::cout << "w" << i << ": " << weights[i] << std::endl;
    }

    // Quick prediction test
    std::cout << "\nQuick predictions:" << std::endl;
    for (size_t i = 0; i < X.size(); ++i) {
        double z = weights[0];
        for (size_t j = 0; j < X[i].size(); ++j) {
            z += weights[j + 1] * X[i][j];
        }
        double prob = sigmoid(z);
        int predicted = (prob >= 0.5) ? 1 : 0;
        std::cout << "Sample " << i << " | True: " << y[i] 
                  << " | Predicted: " << predicted 
                  << " | Probability: " << prob << std::endl;
    }

    return 0;
}