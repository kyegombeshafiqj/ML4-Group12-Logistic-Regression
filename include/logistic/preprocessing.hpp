#pragma once

#include "logistic/types.hpp"
#include <utility>
#include <vector>

namespace logistic {

class Preprocessor {
public:
    // Convert Joshua's Dataset into X and y.
    ProcessedData transform(const Dataset& dataset) const;

    // Split data into training and testing sets.
    std::pair<ProcessedData, ProcessedData> split(
        const ProcessedData& data,
        double train_ratio = 0.8
    ) const;

    // Calculate scaling parameters using training data.
    void fit(const ProcessedData& training_data);

    // Apply the learned scaling parameters to data.
    ProcessedData normalize(const ProcessedData& data) const;

private:
    std::vector<double> means_;
    std::vector<double> standard_deviations_;
};

} // namespace logistic