#include "logistic/statistics.hpp"
#include "logistic/logistic_model.hpp"
#include "logistic/types.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

int main() {
    using namespace logistic;

    // --------------------------------------------------
    // TEST 1: Create a simple dataset
    // --------------------------------------------------

    ProcessedData data;

    data.X = {
        {-1.0},
        { 1.0},
        {-2.0},
        { 2.0}
    };

    data.y = {0, 1, 0, 1};

    // --------------------------------------------------
    // TEST 2: Create a model with zero parameters
    // --------------------------------------------------

    LogisticModel model(1);

    model.setParameters(
        std::vector<double>{0.0},
        0.0
    );

    // --------------------------------------------------
    // TEST 3: Prepare parameters for StatisticsAnalyzer
    //
    // The statistics implementation expects:
    // parameters[0] = intercept/bias
    // parameters[1] = coefficient for feature 1
    // --------------------------------------------------

    std::vector<double> parameters;

    parameters.push_back(model.getBias());

    const std::vector<double>& modelWeights =
        model.getWeights();

    parameters.insert(
        parameters.end(),
        modelWeights.begin(),
        modelWeights.end()
    );

    StatisticsAnalyzer analyzer;

    // --------------------------------------------------
    // TEST 4: Calculate statistical results
    // --------------------------------------------------

    StatisticalAnalysisResult result =
        analyzer.analyze(
            data.X,
            data.y,
            parameters
        );

    // --------------------------------------------------
    // TEST 5: Verify sample count
    // --------------------------------------------------

    assert(result.sample_count == 4);

    // --------------------------------------------------
    // TEST 6: Verify model log-likelihood
    //
    // With zero parameters:
    // p = 0.5 for every sample.
    //
    // Log-likelihood = 4 * ln(0.5)
    // --------------------------------------------------

    const double expectedLogLikelihood =
        4.0 * std::log(0.5);

    assert(
        std::abs(
            result.log_likelihood -
            expectedLogLikelihood
        ) < 1e-9
    );

    // --------------------------------------------------
    // TEST 7: Verify null log-likelihood
    //
    // Two positive labels out of four:
    // p0 = 2/4 = 0.5
    //
    // Null log-likelihood = 4 * ln(0.5)
    // --------------------------------------------------

    const double expectedNullLogLikelihood =
        4.0 * std::log(0.5);

    assert(
        std::abs(
            result.null_log_likelihood -
            expectedNullLogLikelihood
        ) < 1e-9
    );

    // --------------------------------------------------
    // TEST 8: Verify McFadden pseudo-R-squared
    //
    // Both likelihoods are equal:
    // R² = 1 - (LL / LL_null) = 0
    // --------------------------------------------------

    assert(result.pseudo_r2_available);

    assert(
        std::abs(
            result.mcfadden_pseudo_r2
        ) < 1e-9
    );

    // --------------------------------------------------
    // TEST 9: Verify coefficient count
    //
    // One intercept + one feature coefficient = 2
    // --------------------------------------------------

    assert(result.coefficients.size() == 2);

    // Check the intercept name.
    assert(
        result.coefficients[0].name == "intercept"
    );

    // Check the feature coefficient name.
    assert(
        result.coefficients[1].name == "feature_1"
    );

    // Both coefficients should be zero.
    assert(
        std::abs(
            result.coefficients[0].coefficient
        ) < 1e-9
    );

    assert(
        std::abs(
            result.coefficients[1].coefficient
        ) < 1e-9
    );

    // --------------------------------------------------
    // TEST 10: Verify coefficient inference
    //
    // For this dataset, the information matrix is
    // invertible, and both standard errors should exist.
    // --------------------------------------------------

    assert(result.coefficient_inference_available);

    assert(
        result.coefficients[0].standard_error_available
    );

    assert(
        result.coefficients[1].standard_error_available
    );

    // Zero coefficient / positive standard error = z of 0.
    // The two-sided p-value should therefore be 1.
    assert(
        std::abs(
            result.coefficients[0].p_value - 1.0
        ) < 1e-9
    );

    assert(
        std::abs(
            result.coefficients[1].p_value - 1.0
        ) < 1e-9
    );

    // --------------------------------------------------
    // TEST 11: Verify invalid input is rejected
    //
    // X contains four rows, but y contains only three
    // labels. The analyzer should throw an exception.
    // --------------------------------------------------

    bool invalidInputRejected = false;

    try {
        analyzer.analyze(
            data.X,
            std::vector<int>{0, 1, 0},
            parameters
        );
    } catch (const std::invalid_argument&) {
        invalidInputRejected = true;
    }

    assert(invalidInputRejected);

    // --------------------------------------------------
    // TEST 12: Verify inconsistent feature lengths
    // --------------------------------------------------

    bool inconsistentFeaturesRejected = false;

    try {
        analyzer.analyze(
            std::vector<std::vector<double>>{
                {-1.0},
                {1.0, 2.0}
            },
            std::vector<int>{0, 1},
            parameters
        );
    } catch (const std::invalid_argument&) {
        inconsistentFeaturesRejected = true;
    }

    assert(inconsistentFeaturesRejected);

    // --------------------------------------------------
    // TEST 13: Verify invalid labels are rejected
    // --------------------------------------------------

    bool invalidLabelRejected = false;

    try {
        analyzer.analyze(
            data.X,
            std::vector<int>{0, 1, 2, 1},
            parameters
        );
    } catch (const std::invalid_argument&) {
        invalidLabelRejected = true;
    }

    assert(invalidLabelRejected);

    // --------------------------------------------------
    // TEST 14: Verify incorrect parameter count is rejected
    // --------------------------------------------------

    bool invalidParameterCountRejected = false;

    try {
        analyzer.analyze(
            data.X,
            data.y,
            std::vector<double>{0.0}
        );
    } catch (const std::invalid_argument&) {
        invalidParameterCountRejected = true;
    }

    assert(invalidParameterCountRejected);

    // --------------------------------------------------
    // All tests completed
    // --------------------------------------------------

    std::cout
        << "All statistics tests passed successfully."
        << std::endl;

    return 0;
}
