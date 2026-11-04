# Week 4 Progress Report: Preprocessing & Statistical Analysis Modules

**Project:** Machine Learning 4 -- Logistic Regression from Scratch in C++  
**Group:** 12  
**Repository:** `ML4-Group12-logistic-Regression`

---

## 1. Preprocessing Module

**Responsible Member:** Arnold Zziwa Ssebunya  
**Files Involved:** `include/logistic/preprocessing.hpp`, `src/preprocessing.cpp`

### 1.1 Context & Objectives
During the review of `preprocessing.hpp` and `preprocessing.cpp`, focus was placed on two critical data preparation steps:
1. Ensuring feature normalization learns parameters exclusively from the training dataset.
2. Implementing rigorous validation and reproducible logic for the training/testing split.

Separating training and testing data prevents information leakage and ensures that model performance metrics reflect unseen data.

---

### 1.2 Improvement One: Training-Set-Only Normalization (`fit` & `normalize`)

#### Problem
Standardizing features before splitting the dataset or fitting normalization on the whole dataset introduces **data leakage**, as test-set statistics (mean and standard deviation) influence the training pipeline.

#### Implementation
The `fit()` function in `src/preprocessing.cpp` computes feature parameters strictly using `training_data`.

##### Validation & Calculation Steps in `fit()`:
1. Verifies that the training dataset is non-empty.
2. Checks that the row count of features matches the label count.
3. Ensures each sample possesses identical feature counts.
4. Validates that feature values are finite numbers.
5. Computes feature-wise mean ($\mu$) and population standard deviation ($\sigma$).

Parameters are saved in `means_` and `standard_deviations_` inside `include/logistic/preprocessing.hpp`.

##### Standardization Formula:
$$z = \frac{x - \mu}{\sigma}$$

*Where:*
* $x$ is the raw feature value.
* $\mu$ is the training-set mean.
* $\sigma$ is the training-set population standard deviation.
* $z$ is the scaled value.

*Zero-Division Protection:* If a feature's standard deviation is extremely small, it is substituted with `1.0` to avoid division by zero. Constant features evaluate to zero after normalization.

##### Mandatory Execution Sequence:
1. Parse dataset into feature matrix and label vector.
2. Partition dataset into training and testing splits.
3. Execute `fit()` strictly on `training_data`.
4. Call `normalize()` on training set.
5. Call `normalize()` on test set using fitted preprocessor.

---

### 1.3 Improvement Two: Verified Train/Test Splitting (`split`)

#### Implementation & Checks
The `split()` function in `src/preprocessing.cpp` validates the dataset prior to partitioning:
1. Verifies ratio $r \in (0.0, 1.0)$.
2. Verifies matching row count between features and labels.
3. Requires $\ge 2$ samples (prevents empty subsets).
4. Verifies samples contain $\ge 1$ feature.
5. Verifies uniform feature count across all samples.
6. Enforces binary label validation: `0` for benign (B), `1` for malignant (M).

##### Reproducibility
Indices are shuffled using a random number generator seeded with a fixed seed of `42`.

#### Example Split (WDBC Dataset - 569 Samples):
* **Ratio:** 80 / 20 (`0.8`)
* **Training Set:** 455 samples
* **Testing Set:** 114 samples
* **Total:** 569 samples

---

### 1.4 Preprocessing Check Matrix

| Check | Expected Result |
| :--- | :--- |
| Split ratio = `0.8` | ~80% training / ~20% testing samples |
| Re-run split with seed `42` | Identical sample partition |
| Split ratio = `0.0` or `1.0` | Function rejects ratio |
| Feature/label row count mismatch | Function returns error |
| Inconsistent feature dimensions | Function returns error |
| Execute `fit()` | Mean and standard deviation calculated strictly from training set |
| Normalize test set | Reuses training parameters ($\mu, \sigma$) without refitting |

---

## 2. Statistical Analysis Module

**Responsible Member:** Nalukwago Phionah Sharon  
**Files Involved:** `include/logistic/statistics.hpp`, `src/statistics.cpp`, `src/main.cpp`

### 2.1 Context & Objectives
During Week 4, the statistical analysis component was updated to refine statistical calculations, enhance output presentation, and establish appropriate error boundaries regarding coefficient inference limitations.

---

### 2.2 Integration & Model Fit Metrics

The `StatisticsAnalyzer` class was integrated into `main.cpp`. The model's fit improvement relative to a null model is evaluated using McFadden's Pseudo-$R^2$:

$$R^2_{\text{McFadden}} = 1 - \frac{\log L_{\text{model}}}{\log L_{\text{null}}}$$

#### Measured Values:
* **Model Log-Likelihood ($\log L_{\text{model}}$):** $-12.8348$
* **Null Log-Likelihood ($\log L_{\text{null}}$):** $-71.0967$
* **McFadden Pseudo-$R^2$:** $0.819474$

---

### 2.3 Coefficient Statistics & Regularization Constraints

The module calculates:
* Regression coefficients ($\hat{\beta}$)
* Standard errors ($SE(\hat{\beta})$) via information matrix inversion
* Z-statistics
* Approximate two-sided Wald p-values

#### Findings & Limitations:
1. **Multicollinearity:** Strongly correlated features in the WDBC dataset lead to high condition numbers, causing numerical instability during matrix inversion and inflated standard errors.
2. **L2 Regularization Impact:** The standard Fisher Information matrix assumes unregularized logistic regression. Because the model uses L2 regularization, standard Wald p-values should not be taken as un-biased significance metrics.

---

### 2.4 Validation & Error Handling

To prevent numerical issues, `statistics.cpp` performs the following validation steps:
* Confirms dataset non-emptiness.
* Matches feature row counts to label counts.
* Validates feature dimensional consistency.
* Enforces binary label bounds ($y_i \in \{0, 1\}$).
* Checks for finite parameters and feature values.
* Tests matrix invertibility prior to computing variance-covariance metrics.

---

### 2.5 Execution Results

#### Statistical Analysis Summary:
| Metric | Result |
| :--- | :--- |
| Samples Analysed | 114 |
| Model Log-Likelihood | -12.8348 |
| Null Log-Likelihood | -71.0967 |
| McFadden Pseudo-$R^2$ | 0.819474 |

#### Classification Performance Metrics (Test Set):
| Metric | Result |
| :--- | :--- |
| Test Samples | 114 |
| Accuracy | 97.3684% |
| Precision | 1.0000 |
| Recall | 0.916667 |
| Specificity | 1.0000 |
| F1-Score | 0.956522 |

#### Confusion Matrix Breakdown:
* **True Positives (TP):** 33
* **True Negatives (TN):** 78
* **False Positives (FP):** 0
* **False Negatives (FN):** 3

---

## 3. Remaining Action Items & Next Steps

1. **Information Matrix Stability:** Implement matrix conditioning checks or pseudo-inverse routines to handle multicollinearity safely.
2. **L2 Penalty Corrections:** Modify variance-covariance calculations to account for the L2 regularization penalty term.
3. **Availability Indicators:** Check computation success flags prior to displaying Wald stats, masking outputs when matrix inversion is unviable.
4. **Stratified Splitting:** Evaluate stratified sampling during train/test split to guarantee proportional target distribution ($0$ vs $1$) across splits.