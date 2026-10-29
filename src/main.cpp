
#include <iostream>
#include <exception>

#include "logistic/data_loader.hpp"
#include "logistic/preprocessing.hpp"
#include "logistic/logistic_model.hpp"
#include "logistic/trainer.hpp"
#include "logistic/evaluator.hpp"
#include "logistic/statistics.hpp"
int main() {
    try {
        std::cout << "Machine Learning 4 - Logistic Regression\n";

        // 1. Load the dataset
        logistic::DataLoader loader;
        logistic::Dataset dataset =
            loader.loadFromFile("data/wdbc.data");

        std::cout << "Dataset loaded: "
                  << dataset.samples.size() << " samples\n";

        // 2. Convert dataset into features and labels
        logistic::Preprocessor preprocessor;
        logistic::ProcessedData data =
            preprocessor.transform(dataset);

        // 3. Split into training and testing sets
        auto split_result = preprocessor.split(data, 0.8);

         logistic::ProcessedData training_data = split_result.first;
        logistic::ProcessedData testing_data = split_result.second;

        // 4. Learn normalization parameters from training data
        preprocessor.fit(training_data);

        training_data = preprocessor.normalize(training_data);
        testing_data = preprocessor.normalize(testing_data);

        std::cout << "Training samples: "
                  << training_data.X.size() << '\n';
        std::cout << "Testing samples: "
                  << testing_data.X.size() << '\n';

        // 5. Create and train the model
        logistic::LogisticModel model(
            static_cast<int>(training_data.X.at(0).size()));

        logistic::Trainer trainer(model.getConfig());
        trainer.train(model, training_data);

        std::cout << "Training completed.\n";
        std::cout << "Final training loss: "
                  << trainer.getLastLoss() << '\n';

        std::cout << "Model is ready for evaluation.\n";
        std::vector<double> statistical_weights;
statistical_weights.push_back(model.getBias());

for (double weight : model.getWeights()) {
    statistical_weights.push_back(weight);
}
    logistic::StatisticsAnalyzer analyzer;
auto stats = analyzer.analyze(
    testing_data.X,
    testing_data.y,
    statistical_weights
);

std::cout << "\n=== Statistical Analysis ===\n";

std::cout << "Samples analyzed: "
          << stats.sample_count << '\n';

std::cout << "Log-likelihood: "
          << stats.log_likelihood << '\n';

if (stats.pseudo_r2_available) {
    std::cout << "Null log-likelihood: "
              << stats.null_log_likelihood << '\n';

    std::cout << "McFadden Pseudo-R2: "
              << stats.mcfadden_pseudo_r2 << '\n';
} else {
    std::cout << "McFadden Pseudo-R2: unavailable\n";
}

std::cout << "\n=== Coefficient Statistics ===\n";

for (const auto& coefficient : stats.coefficients) {
    std::cout << coefficient.name
              << " | Coefficient: "
              << coefficient.coefficient;

    if (coefficient.standard_error_available) {
        std::cout << " | Standard Error: "
                  << coefficient.standard_error
                  << " | Z-statistic: "
                  << coefficient.z_statistic
                  << " | P-value: "
                  << coefficient.p_value;
    } else {
        std::cout << " | Standard Error: unavailable"
                  << " | Z-statistic: unavailable"
                  << " | P-value: unavailable";
    }

    std::cout << '\n';
}

if (!stats.inference_note.empty()) {
    std::cout << "\nStatistical analysis note: "
              << stats.inference_note << '\n';
}
        // 6. Evaluate the trained model
        logistic::Evaluator evaluator;

        logistic::EvaluationMetrics metrics =
            evaluator.evaluate(model, testing_data);

        std::cout << "\n=== Evaluation Results ===\n";

        std::cout << "Test samples: "
                  << metrics.sample_count << '\n';

        std::cout << "Accuracy: "
                  << metrics.accuracy * 100 << "%\n";

        std::cout << "Precision: "
                  << metrics.precision << '\n';

        std::cout << "Recall: "
                  << metrics.recall << '\n';

        std::cout << "Specificity: "
                  << metrics.specificity << '\n';

        std::cout << "F1-score: "
                  << metrics.f1_score << '\n';

        std::cout << "\nConfusion Matrix:\n";
        std::cout << "True positives: "
                  << metrics.confusion.true_positive << '\n';

        std::cout << "True negatives: "
                  << metrics.confusion.true_negative << '\n';

        std::cout << "False positives: "
                  << metrics.confusion.false_positive << '\n';

        std::cout << "False negatives: "
                  << metrics.confusion.false_negative << '\n';
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }

    return 0;
}