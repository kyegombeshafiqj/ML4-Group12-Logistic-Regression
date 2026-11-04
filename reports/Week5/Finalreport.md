# MAKERERE UNIVERSITY

## SCHOOL OF ENGINEERING 

## COLLEGE OF ENGINEERING, DESIGN, ART AND TECHNOLOGY

## DEPARTMENT OF COMPUTER AND COMMUNICATIONS ENGINEERING

## DEPARTMENT OF ELECTRICAL ENGINEERING

### MACHINE LEARNING 4 – GROUP 12 FINAL PROJECT REPORT

**Project Title:** Implementation of Logistic Regression from Scratch in C++ for Breast Cancer Classification

**Group Members:**

1. KYEGOMBE SHAFIQ .J. - 25/U/08737/PS - BCCE
2. NABUKEERA PROSCOVIA - 25/U/0720 - BBIO
3. MULONDO JOSHUA JORDAN - 25/U/07948/PS - BBIO
4. ZZIWA ARNOLD SSEBUNYA - 25/U/07976/PS - BBIO
5. NALUKWAGO PHIONA SHARON - 25/U/28760/PSA - BCCE
6. KYOMUHENDO SAMALIE - 25/U/08738/PS - BCCE
7. MABANDA ERIYA NESTA - 25/U/24151 - BELE
8. TENDO MALYAMU - 25/U/08786/PS - BCCE



## 1. ABSTRACT

This project involved developing a logistic regression model from scratch in C++ for breast cancer classification using the Wisconsin Diagnostic Breast Cancer (WDBC) dataset. The system classifies tumours as either malignant or benign through data loading, preprocessing, model training, prediction and evaluation. The dataset contains 569 observations with 30 numerical features. In the recorded evaluation, the model achieved an accuracy of 97.37%, demonstrating strong classification performance.

## 2. INTRODUCTION

Breast cancer diagnosis involves identifying whether a tumour is malignant(cancerous) or benign(non-cancerous). Machine learning provides techniques that analyse medical data and identify patterns useful for classification.

The main objective of this project was to implement a multivariable logistic regression model in C++ to classify breast cancer cases.

The specific objectives were to:

* Load and prepare the WDBC dataset.
* Implement logistic regression using C++.
* Train the model using an optimisation algorithm.
* Predict whether a tumour is malignant or benign.
* Evaluate the model using appropriate performance metrics.
* Develop a modular and reusable software implementation.

## 3. METHODOLOGY AND SYSTEM IMPLEMENTATION

### 3.1 Dataset Description

The project used the Wisconsin Diagnostic Breast Cancer dataset, which contains 569 observations. Each observation consists of an identification number, a diagnosis label and 30 numerical features describing characteristics of cell nuclei.

The diagnosis labels were encoded as follows:

* Malignant (M) = 1
* Benign (B) = 0

### 3.2 Logistic Regression Model

Logistic regression is a supervised machine learning algorithm used for binary classification. It predicts the probability that an observation belongs to a particular class.

**Step 1: Linear equation**

The model calculates a weighted sum of the input features:


z = b + (w1*x1) + (w2*x2) + ... + (wn*xn)


Where `z` is the weighted sum, `b` is the bias, `w` represents model weights and `x` represents input features.

**Step 2: Sigmoid function**

The weighted sum is passed through the sigmoid function to obtain a probability between 0 and 1.


P(y = 1) = 1 / (1 + e^(-z))


Here, `P(y = 1)` represents the probability that a tumour is malignant, while `e` is approximately 2.718.

**Step 3: Classification**

The predicted probability determines the classification:


If probability >= 0.5: Malignant (1)

If probability < 0.5: Benign (0)


**Step 4: Training**

During training, the model adjusts its weights and bias to minimise prediction error using an optimisation algorithm.

### 3.3 System Implementation

The project was implemented in C++ using a modular structure to improve organisation, maintainability and reusability.

The main components included:

* **Data Loader:** Reads and loads the WDBC dataset.
* **Preprocessor:** Prepares the dataset for model training.
* **Logistic Model:** Implements the logistic regression mathematical operations.
* **Trainer:** Controls the model training process.
* **Evaluator:** Calculates performance metrics and produces a confusion matrix.
* **Statistics Analyzer:** Computes log-likelihood, McFadden pseudo-R² and coefficient statistics.
* **Main Program:** Coordinates the different components and executes the complete workflow.

Header files defined the interfaces, while source files contained the corresponding implementations. GitHub was used for version control, collaboration and integration of group contributions.

## 4. RESULTS AND DISCUSSION

The recorded program execution successfully loaded 569 observations and divided them into 455 training samples and 114 testing samples. The model was trained and evaluated using the testing dataset.

The following results were obtained:

| Performance Metric | Result |
| ------------------ | -----: |
| Accuracy           | 97.37% |
| Precision          |   100% |
| Recall             | 91.67% |
| Specificity        |   100% |
| F1-score           | 95.65% |

The confusion matrix was:

| Actual / Predicted | Malignant | Benign |
| ------------------ | --------: | -----: |
| Malignant          |        33 |      3 |
| Benign             |         0 |     78 |

The model correctly classified 111 out of 114 observations. It recorded zero false positives and three false negatives, meaning that three malignant cases were incorrectly classified as benign.

The statistical analysis produced a McFadden pseudo-R² value of approximately 0.8195, indicating a substantial improvement in model fit compared with the null model.

## 5. CHALLENGES AND LESSONS LEARNT

During development, the group encountered challenges involving dataset preparation, implementation of mathematical operations in C++, debugging individual modules and integrating contributions from different members.

The project provided practical experience in logistic regression, modular programming, C++ software development, model evaluation and collaborative software development using GitHub.

## 6. CONCLUSION

The project successfully demonstrated the implementation of a logistic regression classification system in C++. Using the WDBC dataset, the model achieved a recorded accuracy of 97.37% on the testing dataset.

The project strengthened our understanding of machine learning algorithms, mathematical modelling, C++ programming and collaborative software development. The modular design also demonstrated how a machine learning system can be organised into reusable components.

## REFERENCES

1. Wolberg, W. H., Street, W. N., & Mangasarian, O. L. (1993). Breast Cancer Wisconsin (Diagnostic) Dataset. UCI Machine Learning Repository.
2. James, G., Witten, D., Hastie, T., & Tibshirani, R. (2021). *An Introduction to Statistical Learning*. Springer.
3. Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer.
4. Stroustrup, B. (2024). Programming: Principles and Practice Using C++ (3rd ed.). Addison-Wesley Professional.

