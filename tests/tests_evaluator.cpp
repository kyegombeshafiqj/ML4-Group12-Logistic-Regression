#include "logistic/evaluator.hpp"
 
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>
 
int main() {
    logistic::LogisticModel model(1);
    model.setParameters(std::vector<double>{1.0}, 0.0);
 
    logistic::ProcessedData test_data;
    test_data.X = {
        {-2.0}, // actual 0, predicted 0 -> TN
        { 2.0}, // actual 1, predicted 1 -> TP
        { 2.0}, // actual 0, predicted 1 -> FP
        {-2.0}  // actual 1, predicted 0 -> FN
    };
    test_data.y = {0, 1, 0, 1};
 
    const logistic::Evaluator evaluator;
    const auto m = evaluator.evaluate(model, test_data);
 
    assert(m.sample_count == 4);
    assert(m.confusion.true_positive == 1);
    assert(m.confusion.true_negative == 1);
    assert(m.confusion.false_positive == 1);
    assert(m.confusion.false_negative == 1);
    assert(std::abs(m.accuracy - 0.5) < 1e-12);
    assert(std::abs(m.precision - 0.5) < 1e-12);
    assert(std::abs(m.recall - 0.5) < 1e-12);
    assert(std::abs(m.specificity - 0.5) < 1e-12);
    assert(std::abs(m.f1_score - 0.5) < 1e-12);
 
    std::cout << "Evaluator tests passed.\\n";
    return 0;
}
