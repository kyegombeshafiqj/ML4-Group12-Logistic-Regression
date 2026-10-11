
#pragma once

#include "logistic/types.hpp"
#include "logistic/logistic_model.hpp"

#include <cstddef>
#include <string>
#include <vector>

namespace logistic {

struct CoefficientStat {
    std::string name;
    double coefficient = 0.0;
    double standard_error = 0.0;
    double z_statistic = 0.0;
    double p_value = 1.0;
    bool standard_error_available = false;
};

struct StatisticalAnalysisResult {
    double log_likelihood = 0.0;
    double null_log_likelihood = 0.0;
    double mcfadden_pseudo_r2 = 0.0;

    std::size_t sample_count = 0;
    std::vector<CoefficientStat> coefficients;

    bool pseudo_r2_available = false;
    bool coefficient_inference_available = false;

    std::string inference_note;
};

class StatisticsAnalyzer {
public:
    StatisticalAnalysisResult analyze(
        const LogisticModel& model,
        const ProcessedData& data
    ) const;

private:
    static double logLikelihood(
        const LogisticModel& model,
        const ProcessedData& data
    );

    static double normalTwoSidedPValue(double z);
};

} 
