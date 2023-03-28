# **Linear Regression: Predictions**

M2U2 - Exercise 4

**What are we going to do?**

- Making predictions with a model

Remember to follow the instructions for the submission of assignments indicated in [Submission Instructions](https://github.com/Tokio-School/Machine-Learning/blob/main/Instrucciones%20entregas.md).

**Task 1: Create a synthetic dataset**

Generate a synthetic dataset for this task In this case, we will not use the dataset to calculate a Θ, the coefficients of the model, but we will use Θ to make predictions about new examples.

In [ ]:

_# TODO: Generate a synthetic dataset, with no error term, in the form of your choice._

m = 0

n = 0

X **=** [**...**]

Theta **=** [**...**]

Y **=** [**...**]

_# Check the values and dimensions (form or "shape") of the vectors_

print('Theta to be estimated:')

print()

print('First 10 rows and 5 columns of X and Y:')

print()

print()

print('Dimensions of X and Y:')

print('shape', 'shape')

**Task 2: Make predictions with the model**

The model consists only of the Θ coefficients, we do not need X or Y to define the model.

Once the optimal Θ has been obtained, either by training the model or by generating the dataset directly as in this case, we will use the linear regression equation to make predictions:

Ypred=X×Θ

In [ ]:

_# TODO: Make predictions using the Theta coefficients_

**def** predict(x, theta):

"""Make predictions for new examples

Positional arguments:

x -- Numpy 2D array with the values of the independent variables from the examples, of size m x n

theta -- Numpy 1D array with the weights of the model coefficients, of size 1 x n (row vector)

Return:

y\_pred -- Numpy 1D array with the dependent/target variable, of size m x 1

"""

y\_pred **=** [**...**]

**return** y\_pred

To make predictions, check that you get a value which is very similar to the original Y:

In [ ]:

_# TODO: Check the difference between your predictions and the original Y, i.e., the residuals_

y\_pred **=** predict(x, theta)

_# Calculate the difference in absolute value between Y\_pred and the original Y_

residuals **=** [**...**]

print(residuals)

To create multiple subgraphs, do you dare to use Matplotlib's subplots function?

[https://matplotlib.org/3.1.0/gallery/subplots\_axes\_and\_figures/subplots\_demo.html](https://matplotlib.org/3.1.0/gallery/subplots_axes_and_figures/subplots_demo.html)

In [ ]:

_# TODO: Graphically represent the predictions and the original Y_

plt **.** figure()

_# Represents a dot plot with original Y vs. X, and predicted Y vs. X, in different colours,_

_# with labels for each colour_

plt **.** title('Predictions')

plt **.** xlabel('X')

plt **.** ylabel('Y real')

plt **.** plot([**...**])

plt **.** grid()

plt **.** show()

In this simple way we can make predictions with a model.

In the following labs we will see how to make predictions for models that we train ourselves.

**Predictions about a dataset with error term**

On the other hand, these predictions fit well with the original Y values, as the dataset did not have any error or variance terms.

_What if you go back to your synthetic dataset generation code and modify it to add an error term? In this case, how will the predictions behave? Will they match the original Y?_

You can check this by modifying the original code cell, adding an error term and rerunning the rest.
