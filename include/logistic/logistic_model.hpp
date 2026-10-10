#pragma once

#include "logistic/types.hpp"

#include <vector>

namespace logistic {

class LogisticModel {
public:
    // Create a model with a specified number of features.
    explicit LogisticModel(
        int n_features,
        const ModelConfig& config = ModelConfig{}
    );

    // Calculate the probability that a sample belongs to class 1.
    double predict_proba(
        const std::vector<double>& features
    ) const;

    // Predict the class: 0 or 1.
    int predict(
        const std::vector<double>& features
    ) const;

    // Predict classes for multiple samples.
    std::vector<int> predict(
        const std::vector<std::vector<double>>& X
    ) const;

    // Train using the model's standalone training method.
    // Samalie's Trainer will be the main training interface.
    void fit(const ProcessedData& data);

    // Initialize parameters to zero.
    void initializeZero();

    // Set model parameters after a training update.
    void setParameters(
        const std::vector<double>& weights,
        double bias
    );

    // Access model parameters.
    const std::vector<double>& getWeights() const;
    double getBias() const;
    double getLastLoss() const;

    // Access model configuration.
    const ModelConfig& getConfig() const;

private:
    // Convert the linear score to a probability.
    double sigmoid(double z) const;

    // Calculate the linear score.
    double linearFunction(
        const std::vector<double>& features
    ) const;

    int n_features_;
    ModelConfig config_;
    std::vector<double> weights_;
    double bias_;
    double last_loss_;
};

} // namespace logistic