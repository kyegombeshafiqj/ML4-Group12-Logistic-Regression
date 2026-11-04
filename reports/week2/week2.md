# Week 2 Progress Report — ML4 Logistic Regression from Scratch in C++

**Group:** 12  
**Project:** Multivariable Logistic Regression implemented from scratch in C++  
**Dataset:** Wisconsin Diagnostic Breast Cancer (WDBC) — 569 total samples (30 numerical features, ID, and diagnosis label where `B` = `0` and `M` = `1`)  
**Reporting Period:** Week 2  
**Status:** Core data/model/training workflow integrated; evaluation and statistical analysis modules are works in progress.  
**Compiled by:** **Kyegombe Shafiq .J.** assisted by **Nabukeera Proscovia** and **Mulondo Joshua Jordan**.

---

## 1. Week 2 Objective
The objective for Week 2 was to transition from research and module planning into C++ implementation. This involved connecting data-loading, preprocessing, the logistic-regression model, and training components into a unified working application that can be extended with evaluation and statistical analysis modules.

---

## 2. Work Completed So Far

### 2.1 Data Loading & Preprocessing
* **Data Loading:** The project utilizes the WDBC dataset containing 569 samples. Each record includes an ID, a binary diagnosis label (`B` = `0`, `M` = `1`), and 30 numerical features. The data-loading component is fully integrated into the application workflow.
* **Preprocessing Workflow:** The preprocessing pipeline is fully integrated using a strict leakage-free sequence:
  1. Convert loaded records into feature rows ($X$) and labels ($y$).
  2. Split the dataset into training and testing subsets.
  3. Fit normalization parameters **using the training subset only**.
  4. Apply those training-derived parameters to both subsets to prevent data leakage from the test set.

### 2.2 Logistic Regression Model & Training Integration
* **Model Integration:** The logistic-regression model is implemented using shared types and model interfaces rather than separate models per module.
* **Pull Requests & Code Merging:** 
  * **Samalie's Training Work:** Submitted and merged via pull request **#17** (`feature/samalie-training`).
  * **Coordinator's Integration & Fixes:** Submitted and merged via pull request **#18** (`fix/logistic-model-method`), which connected loading, preprocessing, splitting, normalization, and model training in `main.cpp`.
* **Local Run Metrics:** An earlier local test run reported:
  * Total samples loaded: **569**
  * Training samples: **455**
  * Testing samples: **114**
  * Final training loss: approximately **0.0444252**  
  *(Note: These figures require formal re-confirmation via a fresh build and run of the current `main` branch before being treated as final).*

### 2.3 GitHub Workflow
The team continues to use feature branches and pull requests to maintain organized module contributions. All future work is required to branch from the latest `main`, undergo peer review via a pull request, and be approved prior to merging.

---

## 3. Current Project Status Summary

| Component | Status | Notes / Responsible Team Member |
|---|---|---|
| WDBC dataset & label convention | **Complete** | Defined and integrated into application |
| Data loader | **Complete** | Implemented and integrated |
| Preprocessing, splitting & normalization | **Complete** | Implemented and integrated securely |
| Logistic-regression model | **Complete** | Implemented and integrated with training |
| Training implementation | **Complete** | Samalie (PR #17) & Coordinator (PR #18) merged |
| Main application workflow | **Needs Verification** | Integrated; requires a fresh build on current `main` |
| Evaluation module (`evaluator.cpp`) | **In Progress** | Assigned to **Tendo** |
| Statistical analysis module (`statistics.cpp`) | **In Progress** | Assigned to **Nesta** |
| End-to-end evaluation & statistics integration | **Pending** | Awaiting module completions |
| Final documentation & comprehensive testing | **Pending** | Planned for post-integration |

---

## 4. Work in Progress (Detailed Module Breakdown)

### 4.1 Evaluation Module — Tendo
* **Planned Files:** `include/logistic/evaluator.hpp`, `src/evaluator.cpp`, `tests/test_evaluator.cpp`
* **Objective:** Evaluate predictions on held-out test data using existing `ProcessedData` and `LogisticModel` interfaces (avoiding duplicate data structures or retraining).
* **Metrics to Output:** Confusion matrix (TP, TN, FP, FN), Accuracy, Precision, Recall/Sensitivity, Specificity, and F1-score.

### 4.2 Statistical Analysis Module — Nesta
* **Planned Files:** `include/logistic/statistics.hpp`, `src/statistics.cpp`, `tests/test_statistics.cpp`
* **Objective:** Compute statistical significance and model fit metrics safely, handling invalid data and singular/unstable information matrices.
* **Metrics to Output:** Model log-likelihood, Null-model log-likelihood, McFadden's pseudo-$R^2$, coefficient standard errors, Wald statistics, and approximate two-sided p-values (with regularized model limitations properly documented).

---

## 5. Issues and Risks Identified
1. **Integration Compatibility:** Because Samalie's training work and the coordinator's corrections were merged through separate pull requests, the current `main` branch must be rebuilt and tested to verify overall consistency.
2. **Unverified Current Build:** While the earlier local run was successful, a fresh build from the latest `main` branch is mandatory before finalizing metrics.
3. **In-Progress Modules:** Tendo's evaluation and Nesta's statistical analysis are still under development, meaning the full end-to-end pipeline is not yet complete.
4. **Shared-File Conflicts:** Both modules will require output integration inside `main.cpp`, requiring careful coordination by the coordinator to prevent conflicting edits.
5. **Testing Gap:** Training loss alone does not validate clinical or classification performance; rigorous testing on held-out test data is still required.

---

## 6. Immediate Next Steps
1. Rebuild and run the latest `main` branch, log any compiler errors, and confirm data sample counts.
2. **Tendo** to continue developing the evaluation module on `feature/tendo-evaluation`, run unit tests, and submit a pull request upon completion.
3. **Nesta** to continue developing the statistical-analysis module on `feature/nesta-statistics`, run unit tests, and submit a pull request upon completion.
4. The coordinator to review incoming pull requests against existing interfaces and manage `main.cpp` integration.
5. Execute the fully integrated application, verify evaluation and statistical outputs against ground truth, and update project documentation with final verified results.

---

**Reporting Note:** This report reflects the status at the time of preparation. Build statements, evaluation metrics, and statistical outputs will be updated in the final repository release once all modules are fully tested and merged.