#include "logistic/logistic_model.hpp"
#include "logistic/trainer.hpp"

#include <cmath>
#include <stdexcept>
#include <algorithm>

namespace logistic {

LogisticModel::LogisticModel(
    int n_features,
    const ModelConfig& config
)
    : n_features_(n_features),
      config_(config),
      weights_(n_features > 0
                   ? static_cast<std::size_t>(n_features)
                   : 0,
               0.0),
      bias_(0.0),
      last_loss_(0.0) {

    if (n_features_ <= 0) {
        throw std::invalid_argument(
            "Number of features must be positive."
        );
    }
}

double LogisticModel::sigmoid(double z) const {
    // Numerically stable sigmoid calculation.
    if (z >= 0.0) {
        return 1.0 / (1.0 + std::exp(-z));
    }

    const double exp_z = std::exp(z);
    return exp_z / (1.0 + exp_z);
}

double LogisticModel::linearFunction(
    const std::vector<double>& features
) const {
    if (features.size() != weights_.size()) {
        throw std::invalid_argument(
            "Feature count does not match model."
        );
    }

    double z = bias_;

    for (std::size_t j = 0; j < weights_.size(); ++j) {
        if (!std::isfinite(features[j])) {
            throw std::invalid_argument(
                "Features must be finite."
            );
        }

        z += weights_[j] * features[j];
    }

    return z;
}

double LogisticModel::predict_proba(
    const std::vector<double>& features
) const {
    return sigmoid(linearFunction(features));
}

// Method name used by Samalie's Trainer.
double LogisticModel::predictProbability(
    const std::vector<double>& features
) const {
    return predict_proba(features);
}

int LogisticModel::predict(
    const std::vector<double>& features
) const {
    return predict_proba(features) >= 0.5 ? 1 : 0;
}

std::vector<int> LogisticModel::predict(
    const std::vector<std::vector<double>>& X
) const {
    std::vector<int> predictions;
    predictions.reserve(X.size());

    for (const auto& features : X) {
        predictions.push_back(predict(features));
    }

    return predictions;
}

void LogisticModel::initializeZero() {
    std::fill(weights_.begin(), weights_.end(), 0.0);
    bias_ = 0.0;
    last_loss_ = 0.0;
}

void LogisticModel::setParameters(
    const std::vector<double>& weights,
    double bias
) {
    if (weights.size() != weights_.size()) {
        throw std::invalid_argument(
            "Weight count does not match model."
        );
    }

    if (!std::isfinite(bias)) {
        throw std::invalid_argument(
            "Bias must be finite."
        );
    }

    for (double weight : weights) {
        if (!std::isfinite(weight)) {
            throw std::invalid_argument(
                "Weights must be finite."
            );
        }
    }

    weights_ = weights;
    bias_ = bias;
}

const std::vector<double>&
LogisticModel::getWeights() const {
    return weights_;
}

double LogisticModel::getBias() const {
    return bias_;
}

double LogisticModel::getLastLoss() const {
    return last_loss_;
}

const ModelConfig& LogisticModel::getConfig() const {
    return config_;
}

// Delegate training to Samalie's Trainer.
void LogisticModel::fit(const ProcessedData& data) {
    Trainer trainer(config_);
    trainer.train(*this, data);
}

} // namespace logistic