//Demonstrates the sigmoid function used in logistic regression

#include <iostream>
#include <cmath>

int main() {
    double z = 2.0;

    double probability = 1.0 / (1.0 + std::exp(-z));

    std::cout << "Logistic Regression Example\n";
    std::cout << "Input (z): " << z << '\n';
    std::cout << "Predicted probability: " << probability << '\n';

    if (probability >= 0.5) {
        std::cout << "Prediction: Malignant\n";
    } else {
        std::cout << "Prediction: Benign\n";
    }

    return 0;
}
