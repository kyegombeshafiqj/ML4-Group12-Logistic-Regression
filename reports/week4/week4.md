# Week 4 Report: Preprocessing Module

**Project:** Machine Learning 4 -- Logistic Regression from Scratch in
C++\
**Group:** 12\
**Module:** Preprocessing\
**Responsible member:** Arnold Zziwa Ssebunya\
**Repository:** `ML4-Group12-logistic-Regression`

## 2. Improvements to the Preprocessing Module

During the review of `preprocessing.hpp` and `preprocessing.cpp`, I
focused on two important parts of data preparation: making sure that
normalization learns from the training data only, and checking that the
training and testing split is performed correctly.

These two steps matter because the model must learn from the training
data without using information from the test data. The test data should
remain separate so that it can give a fair indication of how well the
trained model performs on data it has not learned from.

## 2.1 Improvement One: Use Training Data Only When Calculating Normalization Values

### What needed attention

Before a machine-learning model can use numerical features effectively,
those features may need to be placed on a comparable scale. In this
project, standardization is used. It subtracts the mean from each
feature value and divides the result by that feature's standard
deviation.

A problem can occur if the mean and standard deviation are calculated
using the full dataset before it is divided into training and testing
sets. In that situation, information from the test set influences the
values used to prepare the training data. This is known as **data
leakage**, and it can make the model's evaluation less reliable.

### What was implemented

The `fit()` function in `src/preprocessing.cpp` calculates the mean and
population standard deviation for each feature using the data passed to
it as `training_data`.

For every feature, the function:

1.  Checks that the training data is not empty.
2.  Checks that the number of feature rows matches the number of labels.
3.  Checks that each sample has the same number of features.
4.  Checks that feature values are finite numbers.
5.  Calculates the mean of each feature using the training samples.
6.  Calculates the population standard deviation using those same
    training samples.

The calculated means and standard deviations are stored in `means_` and
`standard_deviations_`, which are declared in
`include/logistic/preprocessing.hpp`.

The `normalize()` function then uses these stored values to standardize
another dataset. This allows the training set and testing set to be
normalized with the same parameters. The test set does not need to
calculate its own mean or standard deviation.

The standardization formula is:

\[ z = `\frac{x - \mu}{\sigma}`{=tex} \]

where:

-   \(x\) is the original feature value;
-   (`\mu`{=tex}) is the mean calculated from the training data;
-   (`\sigma`{=tex}) is the standard deviation calculated from the
    training data;
-   \(z\) is the standardized value.

The code also handles a feature whose standard deviation is extremely
small. It replaces that value with `1.0` so that normalization does not
divide by zero. A constant feature therefore becomes zero after
standardization.

### Why this improvement matters

Using training data only prevents the test set from influencing the
preprocessing values. It also ensures that a new sample can be
normalized using the same values that were learned during training.

For this approach to work correctly, the rest of the project must call
the functions in the right order:

1.  Transform the loaded dataset into features and labels.
2.  Split the data into training and testing sets.
3.  Call `fit()` using the training set only.
4.  Call `normalize()` on the training set.
5.  Call `normalize()` on the testing set using the same fitted
    preprocessor.

Calling `fit()` on the complete dataset or fitting it again on the test
set would defeat this improvement.

## 2.2 Improvement Two: Verify the Training and Testing Split

### What needed attention

The dataset must be divided into two separate groups. The training set
is used to teach the logistic regression model, while the testing set is
kept aside to evaluate the model after training.

The split must preserve the relationship between each sample's features
and its label. It must also account for every sample without placing the
same sample in both sets.

### What was implemented

The `split()` function in `src/preprocessing.cpp` was reviewed and
strengthened with checks before the split takes place.

It now checks that:

1.  The training ratio is greater than `0.0` and less than `1.0`.
2.  The number of feature rows is equal to the number of labels.
3.  At least two samples are available, so neither resulting set has to
    be empty.
4.  Samples contain at least one feature.
5.  All samples have the same number of features.
6.  Labels follow the project's convention: `0` for benign (B) and `1`
    for malignant (M).

After validation, the function creates a list of sample indices and
shuffles that list using a random-number generator with a fixed seed of
`42`. It uses the shuffled indices to place each feature row and its
matching label into either the training set or the testing set.

Using a fixed seed makes the split reproducible. In other words, running
the same split again with the same input and ratio produces the same
arrangement of samples. This makes debugging and comparing model results
easier.

The function calculates the training-set size from the requested ratio.
It also makes sure that both sets contain at least one sample.

### Example of an 80/20 split

For a dataset containing 569 samples, the current implementation
calculates:

-   **Training set:** 455 samples.
-   **Testing set:** 114 samples.
-   **Total:** 569 samples.

The two sets together account for all 569 samples. Each sample's label
is copied alongside its corresponding feature vector, so the features
and labels remain correctly matched.

### Why this improvement matters

A correct split keeps the data used for training separate from the data
used for evaluation. It also makes the experiment repeatable because the
random seed is fixed.

The current implementation performs a random split, but it does not
guarantee that the benign and malignant classes appear in exactly the
same proportions in both sets. If the group later requires that
guarantee, a stratified split can be considered as a further
improvement.

## 2.3 Files Involved

The work for these two improvements concerns the following files:

-   `include/logistic/preprocessing.hpp`
-   `src/preprocessing.cpp`

The header declares the `Preprocessor` class and its functions. The
source file contains the implementation of the transformation,
splitting, fitting and normalization operations.

## 2.4 Checks to Carry Out

The code should be tested to confirm that the improvements work as
intended. The following checks are relevant:

  -----------------------------------------------------------------------
  Check                               Expected result
  ----------------------------------- -----------------------------------
  Split a dataset using a ratio of    Approximately 80% of samples go to
  `0.8`                               training and 20% to testing

  Split the same dataset again        The same split is produced because
                                      the seed is fixed

  Use a ratio of `0.0` or `1.0`       The function rejects the ratio

  Supply different numbers of feature The function reports an error
  rows and labels                     

  Supply samples with different       The function reports an error
  feature counts                      

  Fit using the training set          Means and standard deviations are
                                      calculated from training samples
                                      only

  Normalize the testing set           The stored training means and
                                      standard deviations are reused
  -----------------------------------------------------------------------

These are the expected outcomes of the checks. They should only be
reported as passed after the tests have actually been run.

## 2.5 Conclusion

The first two improvements focus on making data preparation more
reliable before the logistic regression model is trained. The
normalization implementation learns its scaling values from the training
data and reuses them when preparing the testing data. This helps prevent
data leakage. The splitting implementation validates the input, shuffles
sample indices in a repeatable way, and keeps each sample's features
together with its label.

Together, these changes provide a clearer and more dependable process
for preparing the WDBC dataset for model training and evaluation.



Week 4 Progress Report: Statistical Analysis Module

Project: Machine Learning 4 – Logistic Regression from Scratch in C++
Group: 12
Module: Statistical Analysis
Responsible Member: Phionah Sharon
Repository: ML4-Group12-Logistic-Regression

1. Introduction

During Week 4, the group reviewed and improved the statistical analysis component of the Logistic Regression project. The work focused on examining the implementation of statistical calculations, improving the presentation of statistical results, and ensuring that the program communicated the limitations of coefficient inference appropriately.

The review involved examining the "statistics.cpp" and "statistics.hpp" files, together with their integration in "main.cpp".

2. Areas Reviewed and Rectified

2.1 Statistical analysis integration

The integration of the "StatisticsAnalyzer" class into the main program was reviewed. The statistical analysis section was adjusted to improve how the analyzer was initialized and how its results were obtained and presented.

2.2 Log-likelihood and McFadden pseudo-R²

The group reviewed the calculation of the model log-likelihood, null log-likelihood and McFadden pseudo-R².

The implementation uses:

[
R^2_{\text{McFadden}}
=1-\frac{\log L_{\text{model}}}
{\log L_{\text{null}}}
]

The purpose of this measure is to assess the improvement of the fitted logistic regression model over a null model.

The program produced the following results:

- Model log-likelihood: -12.8348
- Null log-likelihood: -71.0967
- McFadden pseudo-R²: 0.819474

These values were checked for consistency with the implemented formula.

2.3 Coefficient statistics

The coefficient analysis was reviewed to examine the calculation of:

- Regression coefficients
- Standard errors
- Z-statistics
- Approximate two-sided p-values

The implementation uses the logistic regression information matrix and matrix inversion to estimate coefficient standard errors.

The review identified that some standard errors were unusually large and that most p-values were close to 1. This may be associated with strong correlations among predictor variables and numerical instability.

Consequently, the group recognised that these coefficient statistics should be interpreted cautiously, particularly because the model uses L2 regularization.

2.4 Validation and error handling

The statistical analysis implementation was reviewed for input validation. Checks include:

- Ensuring the dataset is not empty.
- Confirming that the number of feature rows matches the number of labels.
- Checking that feature rows have consistent dimensions.
- Ensuring that labels are either 0 or 1.
- Checking that feature values and model parameters are finite.
- Checking whether the information matrix can be inverted successfully.

These checks help prevent invalid inputs and numerical problems from producing misleading statistical results.

2.5 Statistical output and interpretation

The statistical output was improved to make the analysis easier to understand. The program reports the number of samples analysed and provides an explanatory note concerning the limitations of approximate Wald p-values and regularization.

The group also identified the importance of checking result-availability indicators before presenting coefficient statistics. Where inference cannot be reliably calculated, the program should communicate that limitation rather than present default values as valid statistical estimates.

3. Results After the Review

The program was executed successfully after the changes, producing the following statistical analysis results:

Measurement| Result
Samples analysed| 114
Log-likelihood| -12.8348
Null log-likelihood| -71.0967
McFadden pseudo-R²| 0.819474

The classification evaluation also produced:

Evaluation metric| Result
Test samples| 114
Accuracy| 97.3684%
Precision| 1.0000
Recall| 0.916667
Specificity| 1.0000
F1-score| 0.956522

The confusion matrix contained 33 true positives, 78 true negatives, zero false positives and three false negatives.

These classification results demonstrate promising predictive performance on the current test split. They are separate from the statistical coefficient-inference results.

4. Challenges Identified

The main challenge identified during the review was the unusually large standard errors and high approximate p-values for several coefficients.

The group recognised that the WDBC dataset contains related measurements, which can introduce multicollinearity and make coefficient estimates difficult to interpret individually.

Another consideration is that the model uses L2 regularization, while the current coefficient-inference calculation is based on an ordinary logistic regression information matrix. Therefore, the reported Wald p-values should not be treated as definitive evidence of statistical significance.

5. Remaining Improvements

The following improvements were identified for further investigation:

1. Further investigate the numerical stability and conditioning of the information matrix.
2. Review the statistical inference method to account appropriately for L2 regularization.
3. Ensure that availability indicators are checked before coefficient statistics are displayed.
4. Clearly distinguish model evaluation on testing data from statistical inference.
5. Add further tests for invalid inputs and cases where coefficient inference is unavailable.

6. Conclusion

The Week 4 review improved the examination and presentation of the statistical analysis component. The program successfully executed and generated log-likelihood, null log-likelihood, McFadden pseudo-R² and coefficient statistics.

The review also identified limitations in the interpretation of standard errors and approximate p-values, which require further investigation.

The group therefore retained the existing logistic regression implementation while identifying targeted improvements to the statistical analysis component rather than unnecessarily rewriting the entire Week 4 Progress Report: Statistical Analysis Module

