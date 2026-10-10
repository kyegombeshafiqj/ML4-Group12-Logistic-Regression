
#include <iostream>
#include <exception>

#include "logistic/data_loader.hpp"
#include "logistic/preprocessing.hpp"
#include "logistic/logistic_model.hpp"
#include "logistic/trainer.hpp"

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
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }

    return 0;
}