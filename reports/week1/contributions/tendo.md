# Week 1 Contribution — Model Evaluation and Prediction

**Name:** TENDO MALYAMU  
**Group:** 12  
**Project:** Machine Learning 4 – Logistic Regression from Scratch  
**Role:** Model Evaluation and Prediction

## My Assigned Area

My part of the project is to study how the logistic regression model
will make predictions and how we can determine whether the model is
performing well.

For Week 1, I focused on understanding and documenting the concepts.
The actual C++ implementation will be done in Week 2.

## What I Researched

### 1. Train and Test Data

I looked at how the dataset can be divided into training and testing
sets.

The training data will be used to teach the logistic regression model,
while the test data will be used later to check how well the model
performs on data that was not used during training.

The group will decide on the final split before implementation.

### 2. Model Prediction

After training, the logistic regression model will produce a
probability for an input sample.

This probability can be converted into a class prediction using a
threshold.

For example, with a threshold of 0.5:

- If the probability is 0.5 or higher, predict class 1.
- If the probability is below 0.5, predict class 0.

This will allow us to compare the model's predictions with the actual
classes in the test dataset.

### 3. Confusion Matrix

I also studied the confusion matrix, which helps us see where the
model's predictions are correct or incorrect.

It consists of:

- **True Positive (TP):** The model correctly predicts the positive
  class.
- **True Negative (TN):** The model correctly predicts the negative
  class.
- **False Positive (FP):** The model predicts positive when the actual
  class is negative.
- **False Negative (FN):** The model predicts negative when the actual
  class is positive.

### 4. Performance Measures

The main evaluation measures I researched are precision, recall and
F1-score.

**Precision**

Precision tells us how reliable the model's positive predictions are.

Precision = TP / (TP + FP)

**Recall**

Recall tells us how many of the actual positive cases the model was
able to identify.

Recall = TP / (TP + FN)

**F1-score**

F1-score combines precision and recall into one measure.

F1 = 2 × (Precision × Recall) / (Precision + Recall)

These measures will help the group evaluate the performance of the
logistic regression model instead of relying only on the number of
correct predictions.

## Planned Workflow

The evaluation process will follow these general steps:

1. Train the logistic regression model using the training data.
2. Give the test data to the trained model.
3. Obtain predicted probabilities.
4. Convert the probabilities into predicted classes.
5. Compare the predicted classes with the actual labels.
6. Construct the confusion matrix.
7. Calculate precision, recall and F1-score.
8. Interpret the results.

## Week 1 Progress

During Week 1, I researched the prediction and evaluation process,
including train/test data, classification thresholds, confusion
matrices, precision, recall and F1-score.

I did not submit C++ implementation during Week 1 because the actual
coding phase will begin in Week 2.

## Next Step

In Week 2, I will implement the prediction and evaluation functions
in C++ and connect them with the trained logistic regression model.

## AI Use

AI tools were used to help me understand the evaluation concepts and
their mathematical formulas. The information was reviewed and used to
prepare for the implementation stage.
