# Week 2 Progress Report — ML4 Logistic Regression from Scratch in C++

**Group:** 12  
**Project:** Multivariable Logistic Regression implemented from scratch in C++  
**Dataset:** Wisconsin Diagnostic Breast Cancer (WDBC)  
**Reporting Period:** Week 2  
**Status:** Core data/model/training workflow integrated; evaluation and statistical analysis modules are works in progress.

## 1. Week 2 Objective

The objective for Week 2 was to move from research and module planning into C++ implementation, connect the data-loading, preprocessing, logistic-regression model and training components, and establish a working application that can be extended with evaluation and statistical analysis.

## 2. Work Completed So Far

### 2.1 Data Loading

The project uses the WDBC dataset, which contains 569 samples. Each record has an ID, a diagnosis label, and 30 numerical features.

The agreed label convention is:

- `B` (benign) = `0`
- `M` (malignant) = `1`

The data-loading component is available in the project and is used by the integrated application to load the dataset.

### 2.2 Preprocessing

The preprocessing component has been integrated into the application workflow. The intended sequence is:

1. Convert the loaded records into feature rows (`X`) and labels (`y`).
2. Split the data into training and testing subsets.
3. Fit normalization parameters using the training subset only.
4. Apply those training-derived parameters to both subsets.

This avoids using information from the test set to calculate normalization parameters.

### 2.3 Logistic Regression Model

The logistic-regression model implementation is present in the repository and is used by the training workflow. The project uses shared types and model interfaces rather than defining a separate model for each module.

### 2.4 Training Implementation and Integration

Samalie's training implementation was submitted through pull request **#17**, from `feature/samalie-training`. The coordinator's integration and correction work was submitted through pull request **#18**, from `fix/logistic-model-method`.

Both pull requests are shown as merged into `main`. The coordinator's changes connected loading, preprocessing, splitting, normalization and model training in `main.cpp`.

The integrated workflow was previously run locally. That run reported:

- Total samples loaded: **569**
- Training samples: **455**
- Testing samples: **114**
- Reported final training loss: approximately **0.0444252**

These values describe the earlier local run. They should be confirmed again by rebuilding and running the current `main` branch before being treated as final verification results.

### 2.5 GitHub Workflow

The team has been using feature branches and pull requests to keep module contributions separate. Future work should remain on each member's own feature branch, based on the latest `main`, and be submitted through a pull request for review before merging.

## 3. Current Project Status

| Component | Status at Report Preparation |
|---|---|
| WDBC dataset and label convention | Defined; dataset loading is used by the application |
| Data loader | Implemented and integrated |
| Preprocessing, splitting and normalization | Implemented and integrated |
| Logistic-regression model | Implemented and integrated with training workflow |
| Training implementation | Samalie's work merged; coordinator's integration and corrections merged |
| Main application workflow | Integrated and previously run locally; current `main` needs a fresh build and test |
| Evaluation module — Tendo | **Work in progress** |
| Statistical analysis module — Nesta | **Work in progress** |
| End-to-end evaluation/statistics integration | Pending |
| Final documentation and comprehensive testing | Pending |

## 4. Work in Progress

### 4.1 Tendo — Evaluation Module

Tendo's evaluation module is a work in progress.

Planned files:

- `include/logistic/evaluator.hpp`
- `src/evaluator.cpp`
- `tests/test_evaluator.cpp`

The module is intended to evaluate predictions on held-out testing data and report:

- Confusion matrix: TP, TN, FP and FN
- Accuracy
- Precision
- Recall/sensitivity
- Specificity
- F1-score

The implementation should use the existing `ProcessedData` and `LogisticModel` interfaces and should not reimplement training or define duplicate shared data structures.

### 4.2 Nesta — Statistical Analysis Module

Nesta's statistical-analysis module is a work in progress.

Planned files:

- `include/logistic/statistics.hpp`
- `src/statistics.cpp`
- `tests/test_statistics.cpp`

The module is intended to calculate:

- Model log-likelihood
- Null-model log-likelihood
- McFadden's pseudo-R²
- Coefficient standard errors and Wald statistics, where numerically available
- Approximate two-sided p-values, with limitations documented

The implementation must handle invalid data and singular or unstable information matrices safely. Conventional Wald p-values from a regularized model should be described as approximate and interpreted cautiously.

## 5. Issues and Risks Identified

1. **Integration compatibility:** Samalie's training work and the coordinator's corrections were merged through separate pull requests. The current `main` branch must be rebuilt and tested to confirm that the combined code is consistent.
2. **Unverified current build:** The earlier successful local run is encouraging, but a fresh build from the latest `main` is still required.
3. **Modules in progress:** Tendo's evaluation and Nesta's statistical analysis are still in progress, so the full pipeline is not yet complete.
4. **Shared-file conflicts:** Both modules may require output integration in `main.cpp`. The coordinator should coordinate these changes to avoid conflicting edits.
5. **Testing gap:** Training loss alone does not demonstrate classification performance. Evaluation on held-out data and module tests remain necessary.

## 6. Immediate Next Steps

1. Rebuild and run the latest `main` branch, record any errors, and confirm the data counts.
2. Tendo should continue the evaluation module on `feature/tendo-evaluation`, test the evaluator, and submit a pull request when ready.
3. Nesta should continue the statistical-analysis module on `feature/nesta-statistics`, test the module, and submit a separate pull request when ready.
4. The coordinator should review both pull requests against the existing interfaces and coordinate any `main.cpp` integration.
5. After both modules are merged, run the full application and verify the evaluation and statistical outputs.
6. Update the README and project report with verified results rather than estimated or assumed metrics.

## 7. Week 2 Conclusion

Week 2 has established the core implementation path for the ML4 logistic-regression project. The data loader, preprocessing workflow, logistic-regression model and Samalie's training implementation have been integrated through pull requests. A previous local run loaded all 569 WDBC samples, split them into 455 training and 114 testing samples, and completed training with a reported loss of approximately 0.0444252.

However, the current combined code still needs a fresh build and verification, while Tendo's evaluation module and Nesta's statistical-analysis module remain works in progress. The next priority is to complete those modules and integrate them into a fully tested application.

---

**Reporting Note:** This report records the status known at preparation time. Update the build and test statements after checking the latest `main` branch, and report evaluation or statistical-analysis results only after the corresponding implementations have been integrated and verified.
The updated repo and report will be availed once evaluation and statistics part are done, tested and implemented.
**Compiled by:** **Kyegombe Shafiq .J.** assisted by **Nabukeera Proscovia** and **Mulondo Joshua Jordan**.
