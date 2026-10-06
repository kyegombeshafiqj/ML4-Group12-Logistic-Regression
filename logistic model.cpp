#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include <iomanip>

// Calculates the linear predictor (z) = beta_0 + dot_product(X, beta)
double calculateLinearPredictor(const std::vector<double>& X, const std::vector<double>& beta, double beta_0) {
    double z = beta_0;
    for (size_t i = 0; i < X.size(); ++i) {
        z += X[i] * beta[i];
    }
    return z;
}

// Maps the linear predictor to a probability between 0 and 1
double sigmoid(double z) {
    return 1.0 / (1.0 + std::exp(-z));
}

// Calculates log-odds: ln(p / (1 - p))
double calculateLogOdds(double p) {
    // Avoid division by zero or log of zero if p is exactly 0 or 1
    if (p <= 0.0 || p >= 1.0) {
        return 0.0; 
    }
    return std::log(p / (1.0 - p));
}

int main() {
    // Set floating-point precision for clean terminal outputs
    std::cout << std::fixed << std::setprecision(4);

    // 1. Define input features (X), weights (beta), and intercept (beta_0)
    std::vector<double> X = {2.5, 1.2, -0.5};
    std::vector<double> beta = {0.8, -0.5, 1.4};
    double beta_0 = 0.2;

    // 2. Compute Linear Predictor (z)
    double z = calculateLinearPredictor(X, beta, beta_0);
    std::cout << "Linear Predictor (z): " << z << "\n";

    // 3. Compute Sigmoid Probability (p)
    double p = sigmoid(z);
    std::cout << "Calculated Probability (p): " << p << "\n";

    // 4. Verify Log-Odds Relationship
    double log_odds = calculateLogOdds(p);
    std::cout << "Log-Odds (logit): " << log_odds << "\n";

    // Check if mathematical relationship holds (Log-Odds == z)
    if (std::abs(log_odds - z) < 1e-9) {
        std::cout << "Verification match: SUCCESS (Log-Odds matches z)\n";
    } else {
        std::cout << "Verification match: FAILED\n";
    }

    return 0;
}
