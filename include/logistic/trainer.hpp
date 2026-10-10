#pragma once

#include "logistic/types.hpp"
#include "logistic/logistic_model.hpp"

namespace logistic {

class Trainer {
public:
    explicit Trainer(const ModelConfig& config = ModelConfig{});

    // Samalie: train Sharon's model using gradient descent.
    void train(
        LogisticModel& model,
        const ProcessedData& training_data
    );

    double getLastLoss() const {
        return last_loss_;
    }

private:
    double computeLoss(
        const LogisticModel& model,
        const ProcessedData& data
    ) const;

    ModelConfig config_;
    double last_loss_ = 0.0;
};

} // namespace logistic