# Machine Learning 4 – Logistic Regression from Scratch
# LOGISTIC REGRESSION FROM SCRATCH IN C++

**Course:** Objected Oriented Programming
**Institution:** Makerere University
**College:** College of Engineering, Design, Art and Technology
**Group:** 12



# 1. PROJECT OVERVIEW

## 1.1 Project Description

This project involves the design and implementation of a multivariable Logistic Regression model from scratch using C++. The system applies machine learning techniques to perform binary classification using the Wisconsin Diagnostic Breast Cancer (WDBC) dataset.

The model analyses numerical features extracted from cell nuclei and predicts whether a tumour is malignant or benign.

The implementation demonstrates the mathematical principles behind logistic regression, including linear combinations of input features, the sigmoid function, model training, probability estimation and classification.

Rather than relying on an existing machine learning library to perform the classification, the project focuses on implementing the essential components of logistic regression in C++.

The project follows a modular software design in which different responsibilities are separated into individual components. This makes the implementation easier to understand, maintain, test and reuse.

## 1.2 Dataset Description

The project uses the Wisconsin Diagnostic Breast Cancer (WDBC) dataset.

The dataset contains:

* 569 observations (samples).
* 30 numerical predictive features describing characteristics of cell nuclei.
* One diagnosis label indicating whether a tumour is malignant or benign.
* One identification field for each observation.

The diagnosis labels are encoded as follows:

| Original Label | Numerical Representation | Meaning   |
| -------------- | -----------------------: | --------- |
| M              |                        1 | Malignant |
| B              |                        0 | Benign    |

The identification field is used to identify observations and is not included among the 30 predictive features.

# 2. PROJECT OBJECTIVES

The main objective is to develop a functional and modular C++ implementation of multivariable logistic regression for binary classification.

The specific objectives are:

1. To understand and implement the mathematical principles underlying logistic regression.
2. To develop a dataset-loading component capable of reading the WDBC dataset.
3. To implement data preprocessing operations required before model training.
4. To develop a logistic regression model that calculates classification probabilities using the sigmoid function.
5. To implement a training mechanism that optimises model parameters.
6. To evaluate classification performance using appropriate statistical and machine learning metrics.
7. To organise the implementation into reusable C++ modules.
8. To apply software engineering practices, including modular programming, testing, documentation and GitHub collaboration.

# 3. MAIN FEATURES

The implementation provides the following features:

* **Dataset Loading:** Reads and loads the WDBC dataset.
* **Data Preprocessing:** Prepares input features and target labels for model training.
* **Multivariable Logistic Regression:** Uses multiple input features to estimate the probability of a sample belonging to a particular class.
* **Model Training:** Adjusts model parameters to minimise prediction error.
* **Classification:** Generates predictions for malignant and benign tumour classes.
* **Performance Evaluation:** Calculates accuracy, precision, recall, specificity and F1-score.
* **Confusion Matrix:** Summarises correct and incorrect classifications.
* **Statistical Analysis:** Computes log-likelihood, null log-likelihood, McFadden pseudo-R² and coefficient statistics.
* **Modular Architecture:** Separates the implementation into components with distinct responsibilities.

# 4. TECHNOLOGIES AND TOOLS

The project uses the following technologies and development tools:

| Technology / Tool    | Purpose                                          |
| -------------------- | ------------------------------------------------ |
| C++                  | Main programming language                        |
| C++ Standard Library | General-purpose programming functionality        |
| Visual Studio Code   | Code editing and development environment         |
| Git                  | Version control                                  |
| GitHub               | Remote repository hosting and team collaboration |
| MinGW / G++          | C++ compilation environment                      |

# 5. MATHEMATICAL FOUNDATION

## 5.1 Introduction to Logistic Regression

Logistic regression is a supervised machine learning algorithm used for binary classification. It estimates the probability that an observation belongs to a particular class.

In this project, the two possible classes are malignant and benign.

## 5.2 Linear Equation

The model first calculates a weighted sum of the input features:

```text
z = b + (w1*x1) + (w2*x2) + ... + (wn*xn)
```

Where:

* `z` = weighted sum of the input features.
* `b` = bias or intercept.
* `w1, w2, ..., wn` = model weights.
* `x1, x2, ..., xn` = input features.
* `n` = number of predictive features (30 in this project).

## 5.3 Sigmoid Function

The calculated weighted sum is passed through the sigmoid function to obtain a probability between 0 and 1.

```text
P(y = 1) = 1 / (1 + e^(-z))
```

Where:

* `P(y = 1)` = probability that a tumour is malignant.
* `e` = mathematical constant approximately equal to 2.718.
* `z` = weighted sum calculated by the linear equation.

## 5.4 Classification

The predicted probability is used to determine the classification:

```text
If probability >= 0.5:
    Malignant (1)

If probability < 0.5:
    Benign (0)
```

## 5.5 Model Training

During training, the model adjusts its weights and bias to minimise prediction error using an optimisation algorithm.

The objective is to find model parameters that allow the algorithm to make accurate predictions on previously unseen observations.

# 6. PROJECT STRUCTURE

The project is organised into separate header and source files to support modular development.

The main components include:

| Module / File        | Responsibility                                          |
| -------------------- | ------------------------------------------------------- |
| `types.hpp`          | Defines shared data structures and configuration types. |
| `data_loader.hpp`    | Declares the dataset-loading interface.                 |
| `data_loader.cpp`    | Implements dataset loading.                             |
| `preprocessing.hpp`  | Declares preprocessing functionality.                   |
| `preprocessing.cpp`  | Implements data preprocessing.                          |
| `logistic_model.hpp` | Declares the logistic regression model.                 |
| `model.cpp`          | Contains the model implementation.                      |
| `trainer.hpp`        | Declares model training functionality.                  |
| `trainer.cpp`        | Implements the training process.                        |
| `evaluator.hpp`      | Defines evaluation structures and interfaces.           |
| `statistics.hpp`     | Declares statistical analysis functionality.            |
| `statistics.cpp`     | Implements statistical calculations.                    |
| `tests/`             | Contains test programs for the project components.      |

The actual repository may contain additional files for the main application, build configuration, examples and supporting functionality.

# 7. MODULE DOCUMENTATION

The project follows a modular design in which each component performs a specific task. The modules communicate through shared data structures and interfaces defined in the header files.

## 7.1 Shared Data Structures – `types.hpp`

This header contains common structures used across the project.

The structures include:

* `Sample`: Represents an individual dataset observation.
* `Dataset`: Holds a collection of observations.
* `ProcessedData`: Represents data prepared for model training.
* `ModelConfig`: Stores configuration parameters used by the logistic regression model.

Centralising shared definitions promotes consistency and reduces unnecessary duplication across modules.

## 7.2 Data Loader Module

**Files:** `data_loader.hpp`, `data_loader.cpp`

The Data Loader is responsible for reading the WDBC dataset and making observations available to other components.

Its responsibilities include:

* Opening and reading the dataset file.
* Extracting diagnosis labels and numerical features.
* Converting the original diagnosis labels into numerical representations.
* Organising loaded observations into the project's data structures.

This module provides the initial dataset required by the preprocessing and training stages.

## 7.3 Preprocessing Module

**Files:** `preprocessing.hpp`, `preprocessing.cpp`

The Preprocessing module prepares the loaded dataset for machine learning.

Its responsibilities include:

* Preparing input features.
* Handling target labels.
* Performing the preprocessing operations implemented in the project.
* Producing data in a format suitable for model training.

Preprocessing ensures that the input data is organised consistently before it is passed to the logistic regression model.

## 7.4 Logistic Regression Model

**Files:** `logistic_model.hpp`, `model.cpp`

The Logistic Model implements the mathematical operations required for binary classification.

Its main responsibilities include:

* Calculating the weighted sum of input features.
* Applying the sigmoid function.
* Estimating classification probabilities.
* Producing predictions using the model's classification procedure.

The model uses 30 predictive features to estimate whether an observation belongs to the malignant or benign class.

## 7.5 Trainer Module

**Files:** `trainer.hpp`, `trainer.cpp`

The Trainer manages the model training process.

Its responsibilities include:

* Configuring training parameters.
* Coordinating model training.
* Updating model parameters through the optimisation procedure.
* Monitoring the training process and its loss.

The objective is to adjust the model parameters so that the model learns patterns from the training dataset.

## 7.6 Evaluator Module

**File:** `evaluator.hpp`

The Evaluator assesses the performance of the trained model using predictions from the testing dataset.

It supports the calculation of:

* Accuracy.
* Precision.
* Recall.
* Specificity.
* F1-score.
* Confusion matrix values.

These measurements help determine how effectively the model distinguishes between malignant and benign observations.

## 7.7 Statistical Analysis Module

**Files:** `statistics.hpp`, `statistics.cpp`

The Statistics Analyzer performs additional statistical analysis of the trained model.

Its responsibilities include:

* Calculating log-likelihood.
* Calculating null log-likelihood.
* Computing McFadden pseudo-R².
* Estimating coefficient statistics.
* Calculating standard errors, Z-statistics and approximate Wald p-values.

These statistical measurements provide additional information about model fit and estimated coefficients.

# 8. INSTALLATION AND REQUIREMENTS

## 8.1 Requirements

To work with the project, a development environment capable of compiling C++ programs is required.

The main requirements include:

* A C++ compiler such as GCC/G++.
* A code editor or IDE such as Visual Studio Code.
* Git for version control.
* Access to the project repository.
* The WDBC dataset required for execution.

## 8.2 Obtaining the Project

Clone the project repository from GitHub:

```bash
git clone <repository-url>
```

Navigate into the downloaded project directory:

```bash
cd <project-directory>
```

Replace the placeholders with the actual GitHub repository URL and directory name.

# 9. COMPILATION AND EXECUTION

The project is compiled using the GNU C++ compiler (G++) through the Visual Studio Code PowerShell terminal.

**Compilation:**

powershell
g++ -Iinclude src/*.cpp -o logistic_app.exe

**Execution:**

powershell
.\logistic_app.exe

The compilation command generates the executable file `logistic_app.exe`, which is then used to run the application.


# 10. USING THE LIBRARY

The library is organised into components that can be incorporated into a C++ application through their respective header files.

A typical application workflow involves:

1. Including the required module headers.
2. Loading the dataset.
3. Preparing the data.
4. Configuring the logistic regression model.
5. Training the model.
6. Generating predictions.
7. Evaluating the predictions.
8. Performing statistical analysis where required.

The modular design allows different components to be accessed through their declared interfaces rather than placing the entire implementation in one source file.

The exact class constructors and function calls should be taken from the public interfaces in the project's header files.

# 11. EXAMPLES AND EXPECTED OUTPUT

The main application demonstrates how the different components work together to perform breast cancer classification.

A recorded execution produced the following output summary:

| Description           |    Result |
| --------------------- | --------: |
| Total dataset samples |       569 |
| Training samples      |       455 |
| Testing samples       |       114 |
| Final training loss   | 0.0444252 |
| Test accuracy         |  97.3684% |
| Precision             |      100% |
| Recall                |  91.6667% |
| F1-score              |  95.6522% |

The program also generates a confusion matrix and statistical analysis results.

These results demonstrate the complete workflow from dataset loading to model evaluation.

# 12. TESTING

The project contains a dedicated `tests/` directory for organising and executing tests of the implemented components.

Testing helps verify the correctness of individual modules and identify errors during development.

The testing process covers the following general areas:

* Dataset loading and data handling.
* Preprocessing operations.
* Logistic regression calculations.
* Model training and prediction.
* Evaluation metrics.
* Statistical calculations.

The tests should be compiled and executed using the project's configured test build process.

The `tests/` directory contains the individual test programs and should be consulted for their corresponding execution instructions.

# 13. RESULTS AND PERFORMANCE EVALUATION

The recorded program execution successfully loaded all 569 observations and divided them into 455 training samples and 114 testing samples.

The following performance results were obtained:

| Performance Metric | Result |
| ------------------ | -----: |
| Accuracy           | 97.37% |
| Precision          |   100% |
| Recall             | 91.67% |
| Specificity        |   100% |
| F1-score           | 95.65% |

## 13.1 Confusion Matrix

The confusion matrix was:

| Actual / Predicted | Malignant | Benign |
| ------------------ | --------: | -----: |
| Malignant          |        33 |      3 |
| Benign             |         0 |     78 |

The model correctly classified 111 out of 114 observations.

It recorded:

* True Positives (TP): 33
* True Negatives (TN): 78
* False Positives (FP): 0
* False Negatives (FN): 3

The model recorded no false positives and three false negatives. This means that three malignant cases were incorrectly classified as benign.

## 13.2 Statistical Analysis

The statistical analysis produced the following recorded values:

| Statistical Measure |   Result |
| ------------------- | -------: |
| Log-likelihood      | -12.8348 |
| Null log-likelihood | -71.0967 |
| McFadden pseudo-R²  | 0.819474 |

The McFadden pseudo-R² indicates a substantial improvement in model fit compared with the null model.

The coefficient statistics also include standard errors, Z-statistics and approximate Wald p-values. These should be interpreted with consideration of the model's regularisation.

# 14. LIMITATIONS

Although the implementation achieved strong performance on the recorded testing dataset, several limitations should be considered:

1. The model was evaluated using a relatively small dataset of 569 observations.
2. The model is designed for binary classification and does not directly support multiclass classification.
3. The performance results are specific to the WDBC dataset and may not generalise to other datasets.
4. The model's predictions should not be treated as a replacement for professional medical diagnosis.
5. Statistical coefficient tests are approximate, particularly when regularisation is applied.

# 15. GROUP CONTRIBUTIONS

The project was developed collaboratively, with each member assigned specific responsibilities across the different modules and project activities.

| Group Member            | Contribution / Responsibility                                                      |
| ----------------------- | ---------------------------------------------------------------------------------- |
| KYEGOMBE SHAFIQ J       | Project coordination, testing, system integration and overall project supervision. |
| NABUKEERA PROSCOVIA     | Project coordination and technical documentation.                                  |
| MULONDO JOSHUA JORDAN   | Data Loader module implementation.                                                 |
| ZZIWA ARNOLD SSEBUNYA   | Data Preprocessing module implementation.                                          |
| NALUKWAGO PHIONA SHARON | Logistic Regression Model implementation.                                          |
| KYOMUHENDO SAMALIE      | Model training module implementation.                                              |
| TENDO MALYAMU           | Model evaluation module implementation.                                            |
| MABANDA ERIYA NESTA     | Statistical Analysis module implementation.                                        |

All members contributed to the overall development process through collaboration, discussions, problem-solving and integration of their respective components.


The final contribution record should reflect the actual work completed by each member.

# 16. TEAM COLLABORATION AND VERSION CONTROL

Git and GitHub were used to support collaborative software development.

The team used version control to organise contributions, maintain the project repository and integrate different modules.

The collaboration process involved:

* Sharing the project repository with group members.
* Assigning different components to individual contributors.
* Making changes to project files.
* Committing and pushing contributions.
* Reviewing and integrating changes.
* Resolving errors arising during development and integration.

This workflow helped the group coordinate development and maintain a shared version of the project.

# 17. CONCLUSION

This project demonstrates the implementation of multivariable logistic regression from scratch using C++. It combines mathematical modelling, data processing, model training, classification, statistical analysis and performance evaluation in a modular software system.

The project achieved a recorded classification accuracy of 97.37% on the testing dataset.

It provides practical experience in machine learning principles and C++ software engineering, while demonstrating the importance of modular programming, testing, documentation and collaborative development.

# 18. REFERENCES

1. Wolberg, W. H., Street, W. N., & Mangasarian, O. L. (1993). Breast Cancer Wisconsin (Diagnostic) Dataset. UCI Machine Learning Repository.
2. James, G., Witten, D., Hastie, T., & Tibshirani, R. (2021). *An Introduction to Statistical Learning*. Springer.
3. Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer.
4. Stroustrup, B. (2024). *Programming: Principles and Practice Using C++* (3rd ed.). Addison-Wesley Professional.



**End**

