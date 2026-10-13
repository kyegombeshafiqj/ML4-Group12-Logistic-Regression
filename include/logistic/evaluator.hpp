#pragma once
 
#include "logistic/types.hpp"
#include "logistic/logistic_model.hpp"
 
#include <cstddef>
 
namespace logistic {
 
struct ConfusionMatrix {
    std::size_t true_positive = 0;
    std::size_t true_negative = 0;
    std::size_t false_positive = 0;
    std::size_t false_negative = 0;
};
 
struct EvaluationMetrics {
    ConfusionMatrix confusion;
    std::size_t sample_count = 0;
    double accuracy = 0.0;
    double precision = 0.0;
    double recall = 0.0;
    double specificity = 0.0;
    double f1_score = 0.0;
};
 
class Evaluator {
public:
    EvaluationMetrics evaluate(
        const LogisticModel& model,
        const ProcessedData& test_data
    ) const;
 
private:
    static double safeDivide(double numerator, double denominator);
};
 
} // namespace logistic
