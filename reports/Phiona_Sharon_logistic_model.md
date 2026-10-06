
Week 1 Contribution — Sharon

Name NALUKWAGO PHIONA SHARON

Project

Machine Learning 4 — Logistic Regression from Scratch

Assigned Task

Multivariable Logistic Regression Model

Work Completed

I studied and worked on the implementation of the multivariable logistic regression model from scratch in C++.

The model uses multiple input features to predict the probability of an observation belonging to a particular class.

Logistic Regression Model

For multiple features, the linear combination is:

z = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ

The linear output is then passed through the sigmoid function:

P(y = 1) = 1 / (1 + e⁻ᶻ)

The output represents the probability that an observation belongs to class 1.

Coefficients

The model contains:

- β₀ — intercept
- β₁, β₂, ..., βₙ — coefficients for the input features

The coefficients determine how strongly each feature contributes to the prediction.

Implementation

I investigated how the model can be implemented without using a machine-learning library.

The implementation should:

1. Initialize the model coefficients.
2. Calculate the linear combination of the input features.
3. Apply the sigmoid function.
4. Calculate the predicted probability.
5. Calculate the error.
6. Update the coefficients during training.
7. Repeat the process until the model converges or reaches the specified number of iterations.

Challenges

- Understanding how multiple features contribute to the logistic regression equation.
- Understanding how the coefficients are updated during training.
- Translating the mathematical model into C++.

Next Steps

- Implement and test the multivariable logistic regression model in C++.
- Test the model using the selected dataset.
- Work with the other group members to integrate the model with preprocessing and evaluation.

AI Use

AI tools were used to assist in understanding the mathematical formulation and implementation of multivariable logistic regression. The concepts and implementation were reviewed before being incorporated into the project.
