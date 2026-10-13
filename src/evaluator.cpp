#include "logistic/evaluator.hpp"
 
#include <cmath>
#include <stdexcept>
 
namespace logistic {
 
double Evaluator::safeDivide(double numerator, double denominator) {
    return denominator == 0.0 ? 0.0 : numerator / denominator;
}
 
EvaluationMetrics Evaluator::evaluate(
    const LogisticModel& model,
    const ProcessedData& test_data
) const {
    if (test_data.X.empty()) {
        throw std::invalid_argument("Evaluation data is empty.");
    }
    if (test_data.X.size() != test_data.y.size()) {
        throw std::invalid_argument(
            "X and y have different numbers of rows."
        );
    }
 
    EvaluationMetrics result;
    result.sample_count = test_data.X.size();
 
    for (std::size_t i = 0; i < test_data.X.size(); ++i) {
        const auto& features = test_data.X[i];
        const int actual = test_data.y[i];
 
        if (actual != 0 && actual != 1) {
            throw std::invalid_argument("Labels must be 0 or 1.");
        }
        if (features.empty()) {
            throw std::invalid_argument("A sample has no features.");
        }
        for (double value : features) {
            if (!std::isfinite(value)) {
                throw std::invalid_argument(
                    "Features must be finite numbers."
                );
            }
        }
 
        // Use the existing LogisticModel prediction API.
        const int predicted = model.predict(features);
        if (predicted != 0 && predicted != 1) {
            throw std::runtime_error(
                "Model prediction must be 0 or 1."
            );
        }
 
        if (actual == 1 && predicted == 1) {
            ++result.confusion.true_positive;
        } else if (actual == 0 && predicted == 0) {
            ++result.confusion.true_negative;
        } else if (actual == 0 && predicted == 1) {
            ++result.confusion.false_positive;
        } else {
            ++result.confusion.false_negative;
        }
    }
 
    const double tp = static_cast<double>(result.confusion.true_positive);
    const double tn = static_cast<double>(result.confusion.true_negative);
    const double fp = static_cast<double>(result.confusion.false_positive);
    const double fn = static_cast<double>(result.confusion.false_negative);
    const double total = static_cast<double>(result.sample_count);
 
    result.accuracy = safeDivide(tp + tn, total);
    result.precision = safeDivide(tp, tp + fp);
    result.recall = safeDivide(tp, tp + fn);
    result.specificity = safeDivide(tn, tn + fp);
    result.f1_score = safeDivide(
        2.0 * result.precision * result.recall,
        result.precision + result.recall
    );
 
    return result;
}
 
} // namespace logistic
