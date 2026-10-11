

#include "logistic/statistics.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <limits>

namespace logistic {
namespace {

const double EPSILON = 1e-12;
const double PIVOT_TOLERANCE = 1e-10;

double sigmoid(double z) {
    if (!std::isfinite(z)) {
        throw std::invalid_argument("Invalid linear predictor.");
    }

    if (z >= 0.0) {
        return 1.0 / (1.0 + std::exp(-z));
    }

    const double e = std::exp(z);
    return e / (1.0 + e);
}

double clampProbability(double p) {
    return std::max(EPSILON, std::min(1.0 - EPSILON, p));
}

double predictProbability(
    const std::vector<double>& x,
    const std::vector<double>& weights
) {
    double z = weights[0];

    for (std::size_t j = 0; j < x.size(); ++j) {
        z += weights[j + 1] * x[j];
    }

    return sigmoid(z);
}

typedef std::vector<std::vector<double> > Matrix;

bool invertMatrix(const Matrix& input, Matrix& inverse) {
    const std::size_t n = input.size();

    if (n == 0) return false;

    Matrix a = input;
    inverse.assign(n, std::vector<double>(n, 0.0));

    for (std::size_t i = 0; i < n; ++i) {
        if (a[i].size() != n) return false;

        inverse[i][i] = 1.0;

        for (std::size_t j = 0; j < n; ++j) {
            if (!std::isfinite(a[i][j])) return false;
        }
    }

    for (std::size_t col = 0; col < n; ++col) {
        std::size_t pivot = col;

        for (std::size_t row = col + 1; row < n; ++row) {
            if (std::abs(a[row][col]) >
                std::abs(a[pivot][col])) {
                pivot = row;
            }
        }

        if (std::abs(a[pivot][col]) < PIVOT_TOLERANCE) {
            return false;
        }

        std::swap(a[pivot], a[col]);
        std::swap(inverse[pivot], inverse[col]);

        const double divisor = a[col][col];

        for (std::size_t j = 0; j < n; ++j) {
            a[col][j] /= divisor;
            inverse[col][j] /= divisor;
        }

        for (std::size_t row = 0; row < n; ++row) {
            if (row == col) continue;

            const double factor = a[row][col];

            for (std::size_t j = 0; j < n; ++j) {
                a[row][j] -= factor * a[col][j];
                inverse[row][j] -= factor * inverse[col][j];
            }
        }
    }

    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            if (!std::isfinite(inverse[i][j])) return false;
        }
    }

    return true;
}

} // namespace

double StatisticsAnalyzer::logLikelihood(
    const std::vector<std::vector<double> >& X,
    const std::vector<int>& y,
    const std::vector<double>& weights
) {
    double total = 0.0;

    for (std::size_t i = 0; i < X.size(); ++i) {
        const double p = clampProbability(
            predictProbability(X[i], weights));

        total += y[i] * std::log(p)
               + (1 - y[i]) * std::log(1.0 - p);
    }

    return total;
}

double StatisticsAnalyzer::normalTwoSidedPValue(double z) {
    return std::erfc(std::abs(z) / std::sqrt(2.0));
}

StatisticalAnalysisResult StatisticsAnalyzer::analyze(
    const std::vector<std::vector<double> >& X,
    const std::vector<int>& y,
    const std::vector<double>& weights
) const {
    if (X.empty()) {
        throw std::invalid_argument("Dataset is empty.");
    }

    if (X.size() != y.size()) {
        throw std::invalid_argument("X and y row counts differ.");
    }

    const std::size_t m = X.size();
    const std::size_t p = X[0].size();

    if (p == 0 || weights.size() != p + 1) {
        throw std::invalid_argument(
            "Feature or model parameter count is invalid.");
    }

    bool hasZero = false;
    bool hasOne = false;

    for (std::size_t i = 0; i < m; ++i) {
        if (X[i].size() != p) {
            throw std::invalid_argument(
                "Feature rows have inconsistent lengths.");
        }

        if (y[i] != 0 && y[i] != 1) {
            throw std::invalid_argument("Labels must be 0 or 1.");
        }

        hasZero = hasZero || y[i] == 0;
        hasOne = hasOne || y[i] == 1;

        for (std::size_t j = 0; j < p; ++j) {
            if (!std::isfinite(X[i][j])) {
                throw std::invalid_argument(
                    "Features must be finite.");
            }
        }
    }

    for (std::size_t j = 0; j < weights.size(); ++j) {
        if (!std::isfinite(weights[j])) {
            throw std::invalid_argument(
                "Model parameters must be finite.");
        }
    }

    StatisticalAnalysisResult result;
    result.sample_count = m;
    result.log_likelihood = logLikelihood(X, y, weights);

    if (hasZero && hasOne) {
        const double positiveFraction =
            static_cast<double>(
                std::count(y.begin(), y.end(), 1)) / m;

        const double p0 = clampProbability(positiveFraction);

        for (std::size_t i = 0; i < m; ++i) {
            result.null_log_likelihood += y[i] == 1
                ? std::log(p0)
                : std::log(1.0 - p0);
        }

        if (result.null_log_likelihood < 0.0 &&
            std::isfinite(result.null_log_likelihood)) {
            result.mcfadden_pseudo_r2 =
                1.0 - result.log_likelihood /
                      result.null_log_likelihood;

            result.pseudo_r2_available =
                std::isfinite(result.mcfadden_pseudo_r2);
        }
    } else {
        result.inference_note =
            "Both outcome classes are required for McFadden pseudo-R2.";
    }

    const std::size_t q = p + 1;
    Matrix information(q, std::vector<double>(q, 0.0));

    for (std::size_t i = 0; i < m; ++i) {
        const double probability =
            clampProbability(predictProbability(X[i], weights));

        const double variance = probability * (1.0 - probability);

        std::vector<double> row(q, 1.0);

        for (std::size_t j = 0; j < p; ++j) {
            row[j + 1] = X[i][j];
        }

        for (std::size_t r = 0; r < q; ++r) {
            for (std::size_t c = 0; c < q; ++c) {
                information[r][c] +=
                    variance * row[r] * row[c];
            }
        }
    }

    Matrix covariance;
    const bool invertible = invertMatrix(information, covariance);
    bool allAvailable = invertible;

    for (std::size_t j = 0; j < q; ++j) {
        CoefficientStat stat;
        stat.name = j == 0
            ? "intercept"
            : "feature_" + std::to_string(j);
        stat.coefficient = weights[j];

        if (invertible && covariance[j][j] > 0.0 &&
            std::isfinite(covariance[j][j])) {
            const double se = std::sqrt(covariance[j][j]);
            const double z = weights[j] / se;
            const double pv = normalTwoSidedPValue(z);

            if (std::isfinite(se) && std::isfinite(z) &&
                std::isfinite(pv)) {
                stat.standard_error = se;
                stat.z_statistic = z;
                stat.p_value = pv;
                stat.standard_error_available = true;
            }
        }

        if (!stat.standard_error_available) {
            allAvailable = false;
        }

        result.coefficients.push_back(stat);
    }

    result.coefficient_inference_available = allAvailable;

    if (!allAvailable) {
        if (!result.inference_note.empty()) {
            result.inference_note += " ";
        }

        result.inference_note +=
            "Coefficient standard errors may be unavailable "
            "because the information matrix is singular or unstable.";
    }

    if (!result.inference_note.empty()) {
        result.inference_note += " ";
    }

    result.inference_note +=
        "Wald p-values are approximate; regularization affects interpretation.";

    return result;
}

} // namespace logistic
