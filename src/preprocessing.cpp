#include "logistic/preprocessing.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <stdexcept>

namespace logistic {

// Convert the loaded WDBC dataset into feature matrix X and labels y.
ProcessedData Preprocessor::transform(const Dataset& dataset) const {
    if (dataset.samples.empty()) {
        throw std::invalid_argument(
            "Preprocessor: Cannot transform an empty dataset."
        );
    }

    ProcessedData result;
    result.X.reserve(dataset.samples.size());
    result.y.reserve(dataset.samples.size());

    for (const auto& sample : dataset.samples) {
        // WDBC contains exactly 30 numerical features per sample.
        if (sample.features.size() != 30) {
            throw std::invalid_argument(
                "Preprocessor: A sample does not contain 30 features."
            );
        }

        // Our project uses B = 0 and M = 1.
        if (sample.label != 0 && sample.label != 1) {
            throw std::invalid_argument(
                "Preprocessor: Labels must be 0 (B) or 1 (M)."
            );
        }

        result.X.push_back(sample.features);
        result.y.push_back(sample.label);
    }

    return result;
}


// Split the data into training and testing sets.
// A fixed random seed makes the split reproducible.
std::pair<ProcessedData, ProcessedData> Preprocessor::split(
    const ProcessedData& data,
    double train_ratio
) const {
    if (train_ratio <= 0.0 || train_ratio >= 1.0) {
        throw std::invalid_argument(
            "Preprocessor: train_ratio must be between 0 and 1."
        );
    }

    if (data.X.size() != data.y.size()) {
        throw std::invalid_argument(
            "Preprocessor: X and y must have the same number of samples."
        );
    }

    const std::size_t n = data.X.size();

    if (n < 2) {
        throw std::invalid_argument(
            "Preprocessor: At least two samples are required for splitting."
        );
    }

    const std::size_t feature_count = data.X[0].size();

    if (feature_count == 0) {
        throw std::invalid_argument(
            "Preprocessor: Samples must contain features."
        );
    }

    for (std::size_t i = 0; i < n; ++i) {
        if (data.X[i].size() != feature_count) {
            throw std::invalid_argument(
                "Preprocessor: Inconsistent feature counts."
            );
        }

        if (data.y[i] != 0 && data.y[i] != 1) {
            throw std::invalid_argument(
                "Preprocessor: Labels must be 0 or 1."
            );
        }
    }

    std::size_t train_size =
        static_cast<std::size_t>(n * train_ratio);

    // Ensure neither set is empty.
    train_size = std::max(
        std::size_t{1},
        std::min(train_size, n - 1)
    );

    std::vector<std::size_t> indices(n);
    std::iota(indices.begin(), indices.end(), 0);

    std::mt19937 generator(42);
    std::shuffle(indices.begin(), indices.end(), generator);

    ProcessedData training;
    ProcessedData testing;

    training.X.reserve(train_size);
    training.y.reserve(train_size);
    testing.X.reserve(n - train_size);
    testing.y.reserve(n - train_size);

    for (std::size_t i = 0; i < n; ++i) {
        const std::size_t index = indices[i];

        if (i < train_size) {
            training.X.push_back(data.X[index]);
            training.y.push_back(data.y[index]);
        } else {
            testing.X.push_back(data.X[index]);
            testing.y.push_back(data.y[index]);
        }
    }

    return {training, testing};
}


// Calculate means and standard deviations from TRAINING DATA ONLY.
void Preprocessor::fit(const ProcessedData& training_data) {
    if (training_data.X.empty()) {
        throw std::invalid_argument(
            "Preprocessor: Cannot fit normalization on empty data."
        );
    }

    if (training_data.X.size() != training_data.y.size()) {
        throw std::invalid_argument(
            "Preprocessor: X and y must have matching sizes."
        );
    }

    const std::size_t n = training_data.X.size();
    const std::size_t feature_count = training_data.X[0].size();

    if (feature_count == 0) {
        throw std::invalid_argument(
            "Preprocessor: Training samples must contain features."
        );
    }

    means_.assign(feature_count, 0.0);
    standard_deviations_.assign(feature_count, 0.0);

    // Validate and calculate feature means.
    for (const auto& row : training_data.X) {
        if (row.size() != feature_count) {
            throw std::invalid_argument(
                "Preprocessor: Inconsistent feature counts in training data."
            );
        }

        for (std::size_t j = 0; j < feature_count; ++j) {
            if (!std::isfinite(row[j])) {
                throw std::invalid_argument(
                    "Preprocessor: Features must be finite numbers."
                );
            }

            means_[j] += row[j];
        }
    }

    for (double& mean : means_) {
        mean /= static_cast<double>(n);
    }

    // Calculate population standard deviations.
    for (const auto& row : training_data.X) {
        for (std::size_t j = 0; j < feature_count; ++j) {
            const double difference = row[j] - means_[j];
            standard_deviations_[j] += difference * difference;
        }
    }

    for (double& stddev : standard_deviations_) {
        stddev = std::sqrt(stddev / static_cast<double>(n));

        // Constant features should not cause division by zero.
        if (stddev < 1e-12) {
            stddev = 1.0;
        }
    }
}


// Apply the previously fitted training statistics to any dataset.
ProcessedData Preprocessor::normalize(
    const ProcessedData& data
) const {
    if (means_.empty() || standard_deviations_.empty()) {
        throw std::logic_error(
            "Preprocessor: Call fit() before normalize()."
        );
    }

    if (data.X.size() != data.y.size()) {
        throw std::invalid_argument(
            "Preprocessor: X and y must have matching sizes."
        );
    }

    ProcessedData normalized = data;

    for (auto& row : normalized.X) {
        if (row.size() != means_.size()) {
            throw std::invalid_argument(
                "Preprocessor: Feature count does not match fitted data."
            );
        }

        for (std::size_t j = 0; j < row.size(); ++j) {
            if (!std::isfinite(row[j])) {
                throw std::invalid_argument(
                    "Preprocessor: Features must be finite numbers."
                );
            }

            row[j] = (row[j] - means_[j])
                     / standard_deviations_[j];
        }
    }

    return normalized;
}

} // namespace logistic