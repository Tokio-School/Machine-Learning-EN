# **Logistic Regression: Cost function**

M2U5 - Exercise 3

**What are we going to do?**

- We will create a synthetic dataset for logistic regression manually, and with Scikit-learn
- We will implement the sigmoid logistic activation function
- We will implement the cost function for logistic regression

Remember to follow the instructions for the submission of assignments indicated in [Submission Instructions](https://github.com/Tokio-School/Machine-Learning/blob/main/Instrucciones%20entregas.md).

In [ ]:

**import** random

**import** numpy **as** np

**Creation a synthetic dataset for logistic regression**

We are going to create a synthetic dataset again, but this time for logistic regression.

We are going to discover how to do it with 2 methods we have used previously: manually, and with Scikit-learn, using the [sklearn\_datasets.make\_classification](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_classification.html) function.

In [ ]:

_# TODO: Manually generate a synthetic dataset with a bias term and an error term_

m = 100

n **=** 2

_# Generate a 2D m x n array with random values between -1 and 1_

_# Insert a first column of 1's as a bias term_

X **=** [**...**]

_# Generate a theta array with n + 1 random values between [0, 1)_

Theta\_true **=** [**...**]

_# Calculate Y as a function of X and Theta\_true_

_# Transform Y to values of 1 and 0 (float) when Y ≥ 0.0_

_# Using a probability as the error term, iterate over Y and change the assigned class to its opposite, 1 to 0, and 0 to 1_

error **=** 0.15

Y **=** [**...**]

Y **=** [**...**]

Y **=** [**...**]

_# Check the values and dimensions of the vectors_

print('Theta and its dimensions to be estimated:')

print()

print()

print('First 10 rows and 5 columns of X and Y:')

print()

print()

print('Dimensions of X and Y:')

print()

In [ ]:

_# TODO: Generate a synthetic dataset with a bias term and an error term with Scikit-learn_

_# Use the same values for m, n, and the error from the previous dataset_

X\_sklearn **=** [**...**]

Y\_sklearn **=** [**...**]

_# Check the values and dimensions of the vectors_

print('First 10 rows and 5 columns of X and Y:')

print()

print()

print('Dimensions of X and Y:')

print()

Since we cannot retrieve the previous coefficients with the Scikit-learn method, we will use the manual method for the rest of the exercise.

**Implement the sigmoid function**

We are going to implement the sigmoid activation function. We will use this function to implement our hypothesis, which transforms the model's predictions to values of 0 and 1.

The sigmoid function:

g(z)=11+e−zY=hθ(x)=g(X×Θ)=11+e−x×ΘT

In [ ]:

_# TODO: Implement the sigmoid activation function_

**def** sigmoid(x, theta):

""" Returns the value of the sigmoid for said x and theta.

Positional arguments:

x -- 1D ndarray with the features of an example

theta -- 1D ndarray with the row or column with the coefficients of the features

Return:

sigmoid -- float (0., 1.) with the value of the sigmoid for these parameters

"""

sigmoid **=** [**...**]

**return** sigmoid

Now plot the result of your function to check its implementation:

In [ ]:

_# TODO: Plot the result of the sigmoid function_

_# For the horizontal axis, use a Z in the linear space [-10, 10] of 100 elements_

_# Plot the values of g(z) as a line and dot plot_

_# Compare the result of the sigmoid with Y, as a dot plot with different coloured dots_

_# For the graph, include a title, legend, grid, and ticks on the relevant vertical axis._

**Implement the cost function**

We are going to implement the non-regularised cost function. This function will be similar to the one we implemented for linear regression in a previous exercise.

Cost function:

Y=hΘ(x)=g(X×ΘT)J(Θ)=−[1m∑i=0m(yilog(hθ(xi))+(1−yi)log(1−hθ(xi))]

In [ ]:

_# TODO: Implement the non-regularised cost function for logistic regression_

**def** logistic\_cost\_function(x, y, theta):

""" Computes the cost function for the considered dataset and coefficients

Positional arguments:

x -- 2D ndarray with the values of the independent variables from the examples, of size m x n

y -- 1D ndarray with the dependent/target variable, of size m x 1 and values of 0 or 1

theta -- 1D ndarray with the weights of the model coefficients, of size 1 x n (row vector)

Return:

j -- float with the cost for this theta array

"""

m **=** [**...**]

_# Remember to check the dimensions of the matrix multiplication to perform it correctly_

j **=** [**...**]

**return** j

As in previous exercises, test your implementation by calculating the cost function for each instance of the dataset.

Check that it returns a float scalar, and not a ndarray. Use np.reshape() for your matrix multiplications, if necessary.

With the correct _theta_, the cost function should be 0. As _theta_ moves away from _Theta\_true_, the cost should increase:

In [ ]:

_# TODO: Test your implementation on the dataset_

theta **=** Theta\_true _# Modify and test several values of theta_

j **=** logistic\_cost\_function(X, Y, theta)

print('Cost of the model:')

print(j)

print('Checked theta and Actual theta:')

print(theta)

print(Theta\_true)
