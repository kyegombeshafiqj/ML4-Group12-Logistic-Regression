// test_evaluator_simple.cpp
// A simple test for the Evaluator.
//
// IDEA: we make a fake dataset where WE already know the right answer,
// then check that the Evaluator gives the same answer.
//
// Our fake model uses 1 feature with weight = 1 and bias = 0.
//   feature = +1  -> model predicts 1 (malignant)
//   feature = -1  -> model predicts 0 (benign)

#include "logistic/evaluator.hpp"
#include "logistic/logistic_model.hpp"
#include "logistic/types.hpp"

#include <iostream>
#include <vector>

using namespace logistic;
using namespace std;

int main() {

    // ---------- Step 1: make the model ----------
    LogisticModel model(1);            // a model with 1 feature

    vector<double> weights;
    weights.push_back(1.0);            // weight = 1
    model.setParameters(weights, 0.0); // bias = 0

    // ---------- Step 2: make two kinds of rows ----------
    vector<double> positive_row;       // a row the model will call class 1
    positive_row.push_back(1.0);

    vector<double> negative_row;       // a row the model will call class 0
    negative_row.push_back(-1.0);

    // ---------- Step 3: build the fake dataset (20 samples) ----------
    ProcessedData data;

    // 8 True Positives: actual = 1, model says 1
    for (int i = 0; i < 8; i++) {
        data.X.push_back(positive_row);
        data.y.push_back(1);
    }

    // 7 True Negatives: actual = 0, model says 0
    for (int i = 0; i < 7; i++) {
        data.X.push_back(negative_row);
        data.y.push_back(0);
    }

    // 2 False Positives: actual = 0, model says 1
    for (int i = 0; i < 2; i++) {
        data.X.push_back(positive_row);
        data.y.push_back(0);
    }

    // 3 False Negatives: actual = 1, model says 0
    for (int i = 0; i < 3; i++) {
        data.X.push_back(negative_row);
        data.y.push_back(1);
    }

    // ---------- Step 4: run the Evaluator ----------
    Evaluator evaluator;
    EvaluationMetrics result = evaluator.evaluate(model, data);

    // ---------- Step 5: print what the Evaluator calculated ----------
    cout << "=== Test 1: known example (8 / 7 / 2 / 3) ===" << endl;
    cout << "TP: " << result.confusion.true_positive << endl;
    cout << "TN: " << result.confusion.true_negative << endl;
    cout << "FP: " << result.confusion.false_positive << endl;
    cout << "FN: " << result.confusion.false_negative << endl;
    cout << "Accuracy:    " << result.accuracy << endl;
    cout << "Precision:   " << result.precision << endl;
    cout << "Recall:      " << result.recall << endl;
    cout << "Specificity: " << result.specificity << endl;
    cout << "F1-score:    " << result.f1_score << endl;
    cout << endl;

    // ---------- Step 6: compare with the answers we calculated by hand ----------
    // Decimal numbers have tiny rounding errors, so we don't use ==.
    // Instead we check that the answer is between a lower and upper value.

    int failed = 0;   // counts how many checks fail

    if (result.confusion.true_positive == 8) cout << "PASS: TP is 8" << endl;
    else { cout << "FAIL: TP should be 8" << endl; failed++; }

    if (result.confusion.true_negative == 7) cout << "PASS: TN is 7" << endl;
    else { cout << "FAIL: TN should be 7" << endl; failed++; }

    if (result.confusion.false_positive == 2) cout << "PASS: FP is 2" << endl;
    else { cout << "FAIL: FP should be 2" << endl; failed++; }

    if (result.confusion.false_negative == 3) cout << "PASS: FN is 3" << endl;
    else { cout << "FAIL: FN should be 3" << endl; failed++; }

    if (result.accuracy > 0.7499 && result.accuracy < 0.7501) cout << "PASS: accuracy is 0.75" << endl;
    else { cout << "FAIL: accuracy should be 0.75" << endl; failed++; }

    if (result.precision > 0.7999 && result.precision < 0.8001) cout << "PASS: precision is 0.8" << endl;
    else { cout << "FAIL: precision should be 0.8" << endl; failed++; }

    if (result.recall > 0.7272 && result.recall < 0.7274) cout << "PASS: recall is 0.7273" << endl;
    else { cout << "FAIL: recall should be 0.7273" << endl; failed++; }

    if (result.specificity > 0.7777 && result.specificity < 0.7779) cout << "PASS: specificity is 0.7778" << endl;
    else { cout << "FAIL: specificity should be 0.7778" << endl; failed++; }

    if (result.f1_score > 0.7618 && result.f1_score < 0.7620) cout << "PASS: F1 is 0.7619" << endl;
    else { cout << "FAIL: F1 should be 0.7619" << endl; failed++; }

    // ---------- Test 2: the model predicts NO positives at all ----------
    // Then precision has 0 on the bottom of the division (TP + FP = 0).
    // Our Evaluator should give 0, not crash or print "nan".
    cout << endl;
    cout << "=== Test 2: model never predicts positive ===" << endl;

    ProcessedData data2;
    for (int i = 0; i < 4; i++) {      // 4 samples that are really 1, model says 0
        data2.X.push_back(negative_row);
        data2.y.push_back(1);
    }
    for (int i = 0; i < 6; i++) {      // 6 samples that are really 0, model says 0
        data2.X.push_back(negative_row);
        data2.y.push_back(0);
    }

    EvaluationMetrics result2 = evaluator.evaluate(model, data2);

    cout << "Precision: " << result2.precision << endl;
    cout << "F1-score:  " << result2.f1_score << endl;

    if (result2.precision == 0.0) cout << "PASS: precision is 0 (no crash)" << endl;
    else { cout << "FAIL: precision should be 0" << endl; failed++; }

    if (result2.f1_score == 0.0) cout << "PASS: F1 is 0 (no crash)" << endl;
    else { cout << "FAIL: F1 should be 0" << endl; failed++; }

    // ---------- Final result ----------
    cout << endl;
    if (failed == 0) {
        cout << "ALL TESTS PASSED" << endl;
    } else {
        cout << failed << " TEST(S) FAILED" << endl;
    }

    return 0;
}