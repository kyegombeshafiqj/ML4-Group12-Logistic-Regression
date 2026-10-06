# Data Preprocessing — Week 1

**Member:** ZZIWA ARNOLD SSEBUNYA  
**Role:** Data Preprocessing Research  
**Project:** Machine Learning 4 – Logistic Regression from Scratch  
**Group:** 12

## Assigned Task

I am responsible for researching and documenting the data preprocessing
component of the logistic regression project.

The actual C++ implementation of the preprocessing module will be
carried out during Week 2.

## 1. Purpose of Data Preprocessing

Data preprocessing prepares the dataset before it is given to the
logistic regression model.

The preprocessing stage is important because the quality and format
of the input data can affect the performance and reliability of the
model.

## 2. Data Cleaning

I researched the steps required to check the dataset for problems
before training.

These include:

- Checking for missing values.
- Checking for invalid or non-numeric values.
- Checking for duplicate or invalid records where applicable.
- Ensuring that the target variable is represented correctly.
- Ensuring that the input features contain valid numerical values.

## 3. Feature and Target Separation

The dataset contains input features and a target variable.

The preprocessing stage will separate:

- **Features (X)** — the input variables used by the model.
- **Target (y)** — the class that the model is expected to predict.

For the WDBC dataset, the diagnosis is the target variable:

- `B` → 0 (Benign)
- `M` → 1 (Malignant)

The remaining numerical measurements are used as features.

## 4. Feature Scaling

I researched feature scaling because the dataset contains features
with different numerical ranges.

A scaling method such as standardization can be used:

z = (x - μ) / σ

where:

- x is the original feature value
- μ is the mean of the feature
- σ is the standard deviation

Scaling can help the optimization process behave more consistently
when gradient-based training is used.

## 5. Training and Test Data

I also researched the importance of separating the dataset into
training and testing data.

The training data is used to learn the model parameters, while the
test data is kept separate and used to evaluate how well the trained
model performs on unseen data.

The exact train/test split will be agreed upon by the group before
implementation.

## 6. Preprocessing Pipeline

The planned preprocessing flow is:

1. Load the dataset.
2. Check the data for invalid or missing values.
3. Separate features from the target.
4. Convert categorical target values into numerical labels.
5. Scale/standardize the features where appropriate.
6. Prepare the processed data for the logistic regression model.
7. Pass the processed data to the training stage.

## 7. Coordination with Other Modules

The preprocessing module will receive data from Joshua's DataLoader
and provide processed features and labels to Sharon's logistic
regression model and the training component.

The interfaces between these modules will be agreed upon before
implementation.

## 8. Week 1 Outcome

During Week 1, I focused on understanding the preprocessing
requirements, data cleaning, feature/target separation, feature
scaling and preparation of data for logistic regression.

No C++ implementation was submitted as part of this Week 1
contribution.

## 9. Next Steps

In Week 2, I will implement the researched preprocessing procedures
in C++ and integrate them with the DataLoader and logistic regression
components.

## 10. AI Use

AI tools were used to assist with understanding data preprocessing
concepts and the mathematical principles behind feature
standardization. The information will be reviewed and verified before
implementation.
