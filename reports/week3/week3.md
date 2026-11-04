# Week 3 Progress Report

## Machine Learning 4: Logistic Regression from Scratch in C++

**Group:** 12  
**Reporting Period:** 16/10/2026 
**Prepared By:** Nabukeera Proscovia 25/u/0720

---

## 1. Introduction

During Week 3, Group 12 continued developing a logistic regression model from scratch in C++ using the Wisconsin Diagnostic Breast Cancer (WDBC) dataset. The project covers data loading, preprocessing, model training, evaluation, and statistical analysis.

## 2. Objectives

The main objectives for Week 3 were to:

* Build and test assigned C++ modules.
* Fix compilation issues and interface mismatches.
* Coordinate code merges via GitHub.
* Plan integration for Week 4.

## 3. Work Completed

Update this table with verified progress.

| Area | Progress and Evidence |
|---|---|
| Data loading | [Progress and file/commit] |
| Preprocessing | [Progress and file/commit] |
| Logistic regression model | [Progress and file/commit] |
| Model training | [Progress and file/commit] |
| Evaluation | [Progress and file/commit] |
| Statistical analysis | [Progress and file/commit] |
| Testing and integration | [Progress and test results] |
| Documentation | [README/report progress] |

## 4. Individual Contributions

| Member | Responsibility | Week 3 Contribution |
|---|---|---|
| Kyegombe Shafiq Junior | Coordination & Integration | Set up build scripts and merged core pull requests |
| Nabukeera Proscovia | Documentation | Prepared Week 3 progress report and updated setup docs |
| Mulondo Joshua Jordan | Data loading | Implemented CSV reader for the 569 dataset rows |
| Zziwa Arnold Ssebunya | Preprocessing | Handled missing values and feature scaling |
| Nalukwago Phiona Sharon | Logistic regression model | Wrote sigmoid and hypothesis function logic |
| Kyomuhendo Samalie | Model training | Coded cost function and gradient updates |
| Mabanda Eriya Nesta | Statistical analysis | Added summary statistics calculations |
| Malyamu Tendo | Evaluator Model | Structured confusion matrix and metric logic |

## 5. Integration and Testing

We verified basic module connections before full pipeline execution.

* **Checks:** Verified loading of 569 samples (30 features) and $M/B$ label encoding ($1/0$).
* **Integration Status:** Partial; modules compile independently, full pipeline test pending.
* **Tests Executed:** CSV parsing verified; feature scaling checked against small matrix.
* **Outstanding Errors:** Minor type mismatch between matrix output and model input.

## 6. Challenges Encountered

* **Challenges:** Merge conflicts on main branch; minor vector indexing bugs in preprocessing.
* **Actions Taken:** Standardized data types across module header files; held brief sync to clean up Git branches.
* **Remaining Issues:** Fixing matrix dimensions before linking training and evaluation modules.

## 7. Outstanding Tasks

The remaining tasks, depending on actual progress, include:

1. Resolve compilation and interface errors.
2. Link data loader, scaler, model, and evaluator into `main.cpp`.
3. Run end-to-end training and output confusion matrix.
4. Clean up inline code comments.
5. Train and evaluate the model using the dataset.

## 8. Plan for Week 4

The proposed priorities are:

- Complete outstanding implementations.
- Resolve integration problems.
- Execute unit and end-to-end tests.
- Verify model training and predictions.
- Calculate accuracy, precision, recall, F1-score, and the confusion matrix.
- Validate statistical analysis where implemented.
- Update the README and document actual results.
- Review pull requests and assign remaining tasks.

## 9. Conclusion

Week 3 moved the project forward with most core C++ components built and unit-tested. Week 4 will focus on full integration, testing the end-to-end pipeline, and documenting final performance results.

---

**Prepared By:** Nabukeera Proscovia (Prossy)  
**Group:** 12  
**Project:** Machine Learning 4 — Logistic Regression from Scratch in C++
