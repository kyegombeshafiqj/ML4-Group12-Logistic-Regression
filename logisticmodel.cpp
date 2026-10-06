Introduction
Logistic regression is a fundamental statistical and machine learning model used primarily for binary classification tasks, where the goal is to predict the probability of an event occurring (e.g., success/failure, fraud/legitimate, or disease/healthy). Unlike linear regression, which outputs continuous values across an infinite range, logistic regression models the relationship between a set of independent features and a bounded probability space between 0 and 1.
The core mechanics of the logistic model rely on three tightly coupled concepts: the linear predictor, the sigmoid function, and the log-odds relationship.
1.1 The Linear Predictor
The foundational layer of the model is the linear predictor (\(z\)). This component combines the input feature vector \(X = [x_1, x_2, \dots, x_n]^T\) with their respective learned weights \(\beta = [\beta_1, \beta_2, \dots, \beta_n]^T\) and an inherent bias or intercept (\(\beta _{0}\)). It represents a standard linear regression equation:
\(z=\beta _{0}+\beta _{1}x_{1}+\beta _{2}x_{2}+\dots +\beta _{n}x_{n}\)
While the linear predictor effectively captures relationships between features, its output range is unbounded, spanning from \(-\infty \) to \(+\infty \). This makes it mathematically unsuitable to be used directly as a raw probability value.
1.2 The Sigmoid Function
To map the unbounded continuous output of the linear predictor into a valid probability score, the model passes \(z\) through a non-linear activation function known as the Sigmoid (or Logistic) function. The sigmoid function (\(\sigma \)) is mathematically formulated as:
\(p=\sigma (z)=\frac{1}{1+e^{-z}}\)
This function squeezes any real-valued number into a strict probability interval of \((0, 1)\). When the linear predictor \(z = 0\), the resulting probability is exactly \(p = 0.5\), which serves as the standard decision threshold for binary classification. As \(z\) approaches positive infinity, \(p\) asymptotically approaches \(1\); conversely, as \(z\) approaches negative infinity, \(p\) asymptotically approaches \(0\).
1.3 The Log-Odds (Logit) Relationship
The underlying geometric beauty of logistic regression lies in its algebraic tractability. By isolating \(z\) from the sigmoid function, we arrive at the logit transformation, which uncovers the ratio of the probability of success to the probability of failure—known as the odds:
\(\text{Odds}=\frac{p}{1-p}\)
Taking the natural logarithm of the odds yields the log-odds (logit):
\(\ln \left(\frac{p}{1-p}\right)=z=\beta _{0}+\beta _{1}x_{1}+\dots +\beta _{n}x_{n}\)
This identity demonstrates that while the relationship between the independent features and the probability \(p\) is non-linear (S-curved), the relationship between the features and the log-odds of the target variable is perfectly linear.
CODE
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
