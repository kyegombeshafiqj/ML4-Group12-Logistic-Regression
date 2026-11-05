# Application Demonstration

## 1. Introduction

This document demonstrates the execution and functionality of the Logistic Regression application developed in C++ for the Machine Learning 4 group project.

The application uses the Wisconsin Diagnostic Breast Cancer (WDBC) dataset to classify tumours as either malignant or benign.

The demonstration covers dataset loading, preprocessing, model training, statistical analysis, model evaluation and standalone examples.

## 2. Running the Application

### Requirements
- C++ compiler (G++).
- WDBC dataset (`data/wdbc.data`).

### Compilation

From the root directory of the repository, compile the application using:

bash
g++ -Iinclude src/*.cpp -o logistic_app.exe

### Execution

Run the compiled application using:

bash
.\logistic_app.exe


## 3. Main Application Demonstration

The main application, implemented in `main.cpp`, integrates the different project modules into one complete machine learning workflow.

### 3.1 Dataset Loading and Preprocessing

The `DataLoader` loads the WDBC dataset, while the `Preprocessor` transforms the data into numerical features and corresponding labels.

The dataset is divided into training and testing sets using an 80:20 split. Normalisation parameters are learned from the training data and applied to both sets.

### 3.2 Model Training

The `LogisticModel` is initialised using the number of input features. The `Trainer` then trains the model using the training dataset.

During training, the model adjusts its parameters to minimise the loss function and learn the relationship between the input features and the target classification.

### 3.3 Statistical Analysis

After training, the `StatisticsAnalyzer` performs statistical analysis using the testing dataset and the trained model parameters.

The analysis includes:
- Log-likelihood and null log-likelihood.
- McFadden Pseudo-R².
- Coefficient estimates.
- Standard errors.
- Z-statistics and p-values where available.

These measurements provide additional insight into the model's statistical performance and the contribution of individual features.

### 3.4 Model Evaluation

The `Evaluator` assesses the trained model using the testing dataset.

The evaluation includes:
- Accuracy.
- Precision.
- Recall.
- Specificity.
- F1-score.
- Confusion matrix.

These metrics help assess how effectively the model distinguishes between malignant and benign cases.

### Main Application Screenshots

The execution and workflow of the main application are illustrated in the following screenshots, located in the `demo` folder:

- **Image 1:** Demonstrates the main application's execution and initial processing stages.
- **Image 2:** Demonstrates the subsequent execution stages, including statistical analysis and model evaluation.

## 4. Standalone Examples

In addition to the main application, two standalone examples demonstrate individual functionalities of the project.

### Example 1: Logistic Regression Prediction

`example01.cpp` demonstrates how logistic regression uses an input value to calculate a predicted probability and assign a classification.

### Example 2: Model Evaluation

`example02.cpp` demonstrates the calculation of classification performance metrics using predefined confusion matrix values.

Both examples were independently compiled and executed successfully.

## 5. Expected Execution Output

After successfully compiling and running the main application, the following output is expected:


Machine Learning 4 - Logistic Regression

Dataset loaded: 569 samples

Training samples: 455
Testing samples: 114

Training completed.
Final training loss: 0.0444252

Model is ready for evaluation.

=== Statistical Analysis ===

Samples analyzed: 114
Log-likelihood: -12.8348
Null log-likelihood: -71.0967
McFadden Pseudo-R2: 0.819474

=== Coefficient Statistics ===

[Intercept and feature coefficient statistics]

Statistical analysis note:
[Inference note concerning approximate Wald p-values
and regularization]

=== Evaluation Results ===

Test samples: 114
Accuracy: 97.3684%
Precision: 1
Recall: 0.916667
Specificity: 1
F1-score: 0.956522

Confusion Matrix:
True positives: 33
True negatives: 78
False positives: 0
False negatives: 3
```

*Note: Coefficient statistics and the inference note are abbreviated above. Their complete values are displayed when the application executes.*

### Standalone Examples Output

**Example 1 – Logistic Regression Prediction**


Logistic Regression Example

Input (Z): 2
Predicted probability: 0.880797
Prediction: Malignant


**Example 2 – Model Evaluation**


Model Evaluation Example

Accuracy: 97.3684%
Precision: 1
Recall: 0.916667
Specificity: 1


### Execution Evidence

A screenshot showing the actual expected execution output printed in the terminal is provided in the `examples simulations` folder within the `demo` directory.

This screenshot provides additional visual evidence of the successful execution of the application and its standalone examples.

## 6. Conclusion

The demonstration confirms that the Logistic Regression application executes successfully and integrates the different modules into a complete machine learning workflow.

The application successfully loads and processes the WDBC dataset, trains the logistic regression model, performs statistical analysis and evaluates its classification performance.

The standalone examples further demonstrate individual prediction and evaluation functionalities.
