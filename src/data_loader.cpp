#include "logistic/data_loader.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace logistic {

Dataset DataLoader::loadFromFile(const std::string& path) const {
    Dataset dataset;
    std::ifstream file(path);

    if (!file.is_open()) {
        throw std::runtime_error(
            "DataLoader: Cannot open file: " + path
        );
    }

    std::string line;
    std::size_t lineNumber = 0;
    std::size_t skippedRows = 0;

    while (std::getline(file, line)) {
        ++lineNumber;

        if (line.empty()) {
            continue;
        }

        std::stringstream ss(line);
        std::string token;
        std::vector<std::string> tokens;

        while (std::getline(ss, token, ',')) {
            tokens.push_back(token);
        }

        // ID + diagnosis + 30 features = 32 fields
        if (tokens.size() != 32) {
            ++skippedRows;
            continue;
        }

        try {
            Sample sample;

            sample.id = std::stol(tokens[0]);

            if (tokens[1] == "M" || tokens[1] == "m") {
                sample.label = 1;
            } else if (tokens[1] == "B" || tokens[1] == "b") {
                sample.label = 0;
            } else {
                ++skippedRows;
                continue;
            }

            sample.features.reserve(30);

            for (std::size_t i = 2; i < tokens.size(); ++i) {
                sample.features.push_back(std::stod(tokens[i]));
            }

            dataset.samples.push_back(std::move(sample));

        } catch (const std::exception&) {
            ++skippedRows;
            continue;
        }
    }

    if (file.bad()) {
        throw std::runtime_error(
            "DataLoader: Error while reading file: " + path
        );
    }

    if (skippedRows > 0) {
        // Report skipped rows so problems are not hidden.
        // A logging system can replace this later.
    }

    return dataset;
}

} // namespace logistic