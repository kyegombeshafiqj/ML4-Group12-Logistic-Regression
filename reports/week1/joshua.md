# Joshua Jordan - Week 1 Progress Report

## 1. Name, Role, Module
- **Name:** Joshua Jordan
- **Group:** Group 12
- **Role:** Data Loading Module Lead (MD ONLY -Week 1)
- **Module Assigned:** Data_loader
- **Branch:** `feature/joshua-data-loading`

## 2. Theory / Understanding

My module is the entry point of the entire logistic regression library. Without it, no other module can work.

**Understanding wdbc.data dataset:**
The dataset is Wisconsin Diagnostic Breast Cancer (WDBC) with 569 samples. Each line has 32 comma-separated values:
- Column 1: ID number (e.g., 842302)
- Column 2: Diagnosis: M = Malignant (1), B = Benign (0)
- Columns 3-32: 30 real-valued features computed from cell nucleus image (mean radius, texture, perimeter, area, smoothness, etc for 10 features x 3: mean, se, worst)

Example line:
`842302,M,17.99,10.38,122.8,1001.0,...`

**Logistic Regression Library Flow:**
`DataLoader (me)` -> `Preprocessing (Arnold)` -> `LogisticModel (Sharon)` -> `Training (Samalie)` -> `Evaluator (Tendo)` + `Statistics (Nesta)`

My output must match Arnold's expected input. We agreed:
- I will provide `Dataset` with `std::vector<Sample>` where each Sample has ID, label (0/1), and 30 features.
- I will also provide helper methods `getFeaturesMatrix()` returns `vector<vector<double>>` and `getLabels()` returns `vector<int>` for easy matrix use.

**Key Design Principles from Lecturer Section 3:**
- Library not program: Provide reusable class, not main()
- Separate interface and implementation: .hpp declares WHAT, .cpp defines HOW
- Handle invalid inputs: File not found, blank lines, wrong column count

## 3. What I Completed - Week 1

Since Week 1 is focused on research, understanding and documentation, no C++ implementation was submitted for this module.
During Week 1, I:
- Studied the structure of the WDBC dataset.
- Confirmed that the dataset contains 569 samples and 32 columns.
- Investigated how the CSV-style data can be parsed in C++ using std::ifstream, std::getline, std::stringstream, and std::stod.
- Studied the data structures that will be required for the DataLoader during Week 2.
- Considered how the DataLoader output will connect to the preprocessing module.
- Researched input validation requirements such as missing files, blank lines and incorrect column counts.

Week 1 Implementation status
No C++ implementation was merged as part of my Week 1 contribution. The implementation will be developed during Week 2.