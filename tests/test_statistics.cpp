

#include "logistic/statistics.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

bool closeEnough(double a, double b, double tolerance = 1e-8) {
    return std::abs(a - b) < tolerance;
}

int main() {
    logistic::StatisticsAnalyzer analyzer;

    std::vector<std::vector<double> > X = {
        {-2.0}, {-1.0}, {1.0}, {2.0}
    };
    std::vector<int> y = {0, 0, 1, 1};
    std::vector<double> weights = {0.0, 0.0};

    logistic::StatisticalAnalysisResult result =
        analyzer.analyze(X, y, weights);

    assert(result.sample_count == 4);
    assert(closeEnough(result.log_likelihood, 4.0 * std::log(0.5)));
    assert(closeEnough(result.null_log_likelihood, 4.0 * std::log(0.5)));
    assert(result.pseudo_r2_available);
    assert(closeEnough(result.mcfadden_pseudo_r2, 0.0));
    assert(result.coefficients.size() == 2);
    assert(result.coefficients[0].name == "intercept");
    assert(result.coefficients[1].name == "feature_1");
    assert(result.coefficient_inference_available);

    for (std::size_t i = 0; i < result.coefficients.size(); ++i) {
        assert(result.coefficients[i].standard_error_available);
        assert(result.coefficients[i].standard_error > 0.0);
        assert(closeEnough(result.coefficients[i].p_value, 1.0));
    }

    std::vector<double> nonzeroWeights = {-0.2, 0.5};
    result = analyzer.analyze(X, y, nonzeroWeights);
    assert(std::isfinite(result.log_likelihood));

    bool threw = false;
    try {
        std::vector<int> invalidY = {0, 2, 1, 1};
        analyzer.analyze(X, invalidY, weights);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);

    threw = false;
    try {
        std::vector<int> shortY = {0, 1};
        analyzer.analyze(X, shortY, weights);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);

    threw = false;
    try {
        std::vector<std::vector<double> > emptyX;
        std::vector<int> emptyY;
        analyzer.analyze(emptyX, emptyY, weights);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);

    std::vector<int> constantY = {1, 1, 1, 1};
    result = analyzer.analyze(X, constantY, weights);
    assert(!result.pseudo_r2_available);

    std::cout << "All statistics tests passed.\n";
    return 0;
}
