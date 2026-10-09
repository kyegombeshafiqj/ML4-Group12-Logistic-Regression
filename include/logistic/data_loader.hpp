#pragma once

#include "logistic/types.hpp"
#include <string>

namespace logistic {

class DataLoader {
public:
    // Loads a dataset from a file and returns it
    Dataset loadFromFile(const std::string& path) const;
};

} // namespace logistic