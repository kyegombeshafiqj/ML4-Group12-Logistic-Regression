#include "logistic/trainer.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

namespace logistic {

Trainer::Trainer(const ModelConfig& config)
    : config_(config) {

    if (config_.learning_rate <= 0.0 ||
        !std::isfinite(config_.learning_rate) ||
        config_.epochs <= 0 ||
        config_.l2_lambda < 0.0 ||
        !std::isfinite(config_.l2_lambda) ||
        config_.batch_size < 0) {
        throw std::invalid_argument(
            "Trainer: Invalid training configuration."
        );
    }
}


// Calculate binary cross-entropy plus L2 regularization.
double Trainer::computeLoss(
    const LogisticModel& model,
    const ProcessedData& data
) const {
    if (data.X.empty() || data.X.size() != data.y.size()) {
        throw std::invalid_argument(
            "Trainer: Invalid data for loss calculation."
        );
    }

    const std::size_t m = data.X.size();
    const double epsilon = 1e-15;

    double loss = 0.0;

    for (std::size_t i = 0; i < m; ++i) {
        const double probability =
            model.predict_proba(data.X[i]);

const double p =
    (probability < epsilon) ? epsilon :
    (probability > 1.0 - epsilon) ? 1.0 - epsilon :
    probability;



        const int label = data.y[i];

        loss -= label * std::log(p)
              + (1 - label) * std::log(1.0 - p);
    }

    loss /= static_cast<double>(m);

    // L2 penalty on weights only, not the bias.
    const std::vector<double> weights = model.getWeights();

    double sum_squared_weights = 0.0;

    for (double weight : weights) {
        sum_squared_weights += weight * weight;
    }

    loss += config_.l2_lambda * sum_squared_weights
          / (2.0 * static_cast<double>(m));

    return loss;
}


// Train the model using gradient descent.
void Trainer::train(
    LogisticModel& model,
    const ProcessedData& training_data
) {
    if (training_data.X.empty() ||
        training_data.X.size() != training_data.y.size()) {
        throw std::invalid_argument(
            "Trainer: X and y must have matching, non-empty samples."
        );
    }

    const std::size_t m = training_data.X.size();
    const std::size_t n_features =
        training_data.X.front().size();

    if (n_features != 30) {
        throw std::invalid_argument(
            "Trainer: WDBC samples must have 30 features."
        );
    }

    for (std::size_t i = 0; i < m; ++i) {
        if (training_data.X[i].size() != n_features) {
            throw std::invalid_argument(
                "Trainer: Inconsistent feature counts."
            );
        }

        if (training_data.y[i] != 0 &&
            training_data.y[i] != 1) {
            throw std::invalid_argument(
                "Trainer: Labels must be 0 or 1."
            );
        }

        for (double value : training_data.X[i]) {
            if (!std::isfinite(value)) {
                throw std::invalid_argument(
                    "Trainer: Features must be finite numbers."
                );
            }
        }
    }

    // Start from zero weights and zero bias.
    model.initializeZero();

    std::vector<std::size_t> indices(m);
    std::iota(indices.begin(), indices.end(), 0);

    // Fixed seed makes training reproducible.
    std::mt19937 generator(42);

    const std::size_t batch_size =
        config_.batch_size == 0
            ? m
            : std::min(
                static_cast<std::size_t>(config_.batch_size), m
              );

    for (int epoch = 0; epoch < config_.epochs; ++epoch) {
        std::shuffle(indices.begin(), indices.end(), generator);

        for (std::size_t start = 0; start < m;
             start += batch_size) {

            const std::size_t end =
                std::min(start + batch_size, m);

            const std::size_t current_batch_size = end - start;

            std::vector<double> grad_w(n_features, 0.0);
            double grad_b = 0.0;

            // Forward pass and gradient accumulation.
            for (std::size_t k = start; k < end; ++k) {
                const std::size_t i = indices[k];

                const double p =
                    model.predict_proba(training_data.X[i]);

                const double error =
                    p - static_cast<double>(training_data.y[i]);

                grad_b += error;

                for (std::size_t j = 0; j < n_features; ++j) {
                    grad_w[j] +=
                        error * training_data.X[i][j];
                }
            }

            // Average gradients over this mini-batch.
            for (double& gradient : grad_w) {
                gradient /=
                    static_cast<double>(current_batch_size);
            }

            grad_b /=
                static_cast<double>(current_batch_size);

            // Add the L2 gradient to each weight.
            const std::vector<double> old_weights =
                model.getWeights();

            for (std::size_t j = 0; j < n_features; ++j) {
                grad_w[j] += config_.l2_lambda * old_weights[j]
                           / static_cast<double>(m);
            }

            // Update weights and bias.
            std::vector<double> new_weights = old_weights;

            for (std::size_t j = 0; j < n_features; ++j) {
                new_weights[j] -=
                    config_.learning_rate * grad_w[j];
            }

            const double new_bias =
                model.getBias()
                - config_.learning_rate * grad_b;

            model.setParameters(new_weights, new_bias);
        }

        // Record loss periodically.
        if (epoch % 100 == 0) {
            last_loss_ = computeLoss(model, training_data);
        }
    }

    // Always record the final training loss.
    last_loss_ = computeLoss(model, training_data);
}

} // namespace logistic