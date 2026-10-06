# Training and Optimization — Week 1

**Member:** KYOMUHENDO SAMALIE 
**Role:** Training and Optimization Research
**Project:** Machine Learning 4 – Logistic Regression from Scratch
**Group:** 12

## Assigned Task

I am responsible for researching and documenting the training and
optimization process that will be used to train the logistic regression
model.

The actual C++ implementation will be carried out during Week 2.

## 1. Logistic Regression Training

Logistic regression learns the model coefficients from the training
data.

The model first calculates a weighted combination of the input
features:

z = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ

The result is passed through the sigmoid function to obtain a
probability:

P(y = 1) = 1 / (1 + e⁻ᶻ)

The model's coefficients are then adjusted during training so that
the predicted probabilities become closer to the actual target
values.

## 2. Loss Function

I researched the loss function used for binary logistic regression.

Binary cross-entropy (log loss) can be expressed as:

L = -[y log(p) + (1-y) log(1-p)]

where:

- y is the actual class
- p is the predicted probability

The loss measures how different the model's predictions are from the
actual labels.

## 3. Gradient Descent

I researched gradient descent as the optimization method that can be
used to update the model coefficients.

The general update rule is:

β = β - α(gradient)

where:

- β represents the model coefficients
- α is the learning rate
- gradient represents the direction and magnitude of the change
  required to reduce the loss

The process is repeated over multiple iterations until the model
reaches the required stopping condition.

## 4. Learning Rate

The learning rate controls how large each coefficient update is.

A learning rate that is too large may cause the training process to
overshoot the minimum, while a learning rate that is too small may
make training very slow.

The appropriate learning rate will therefore need to be selected and
tested during implementation.

## 5. Training Process

The planned training process is:

1. Initialize the model coefficients.
2. Calculate the linear combination of the input features.
3. Apply the sigmoid function.
4. Calculate the prediction error/loss.
5. Calculate the gradients.
6. Update the coefficients.
7. Repeat for the required number of iterations.
8. Stop when the specified stopping condition is reached.

## 6. Week 1 Outcome

During Week 1, I focused on understanding the training process,
loss function, gradient descent and learning-rate requirements for
logistic regression.

No C++ implementation was submitted as part of this Week 1
contribution.

## 7. Next Steps

In Week 2, I will implement the training and optimization process
in C++ and integrate it with the logistic regression model developed
by the group.

## 8. AI Use

AI tools were used to assist in understanding logistic regression
training, gradient descent and the loss function. The concepts will
be reviewed and verified before implementation.