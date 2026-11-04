cmake_minimum_required(VERSION 3.16)

project(ML4_Group12_Logistic_Regression LANGUAGES CXX)

# Use C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Locate all C++ implementation files inside src/
file(GLOB_RECURSE PROJECT_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
)

# Create the application
add_executable(logistic_app ${PROJECT_SOURCES})

# Locate project header files
target_include_directories(logistic_app PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

# Place the executable in the build directory
set_target_properties(logistic_app PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)
