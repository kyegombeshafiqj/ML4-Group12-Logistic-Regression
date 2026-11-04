//Demonstrates how classification performance metrics are calculated

#include <iostream>

int main() {
    int TP = 33;
    int TN = 78;
    int FP = 0;
    int FN = 3;

    double accuracy =
        static_cast<double>(TP + TN) / (TP + TN + FP + FN);

    double precision =
        static_cast<double>(TP) / (TP + FP);

    double recall =
        static_cast<double>(TP) / (TP + FN);

    double specificity =
        static_cast<double>(TN) / (TN + FP);

    std::cout << "Model Evaluation Example\n";
    std::cout << "Accuracy: " << accuracy * 100 << "%\n";
    std::cout << "Precision: " << precision << '\n';
    std::cout << "Recall: " << recall << '\n';
    std::cout << "Specificity: " << specificity << '\n';

    return 0;
}
