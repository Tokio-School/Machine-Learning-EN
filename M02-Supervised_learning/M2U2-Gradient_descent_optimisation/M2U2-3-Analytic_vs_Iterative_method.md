# **Multivariate Linear Regression: Analytic vs Iterative method**

M2U2 - Exercise 3

**What are we going to do?**

- Solve linear regression using the analytical method

Remember to follow the instructions for the submission of assignments indicated in [Submission Instructions](https://github.com/Tokio-School/Machine-Learning/blob/main/Instrucciones%20entregas.md).

**Solve the model using the analytic method**

This time, we will solve or train the model using the normal equation, which has the following form:

Θ=(XT×X)−1×XT×Y

In [ ]:

**import** numpy **as** np

**Task 1: Generate a synthetic dataset with no error term**

In [ ]:

_# TODO: Generate a synthetic dataset, with no error term, in the form of your choice._

m = 0

n = 0

X **=** [**...**]

Theta\_verd **=** [**...**]

Y **=** [**...**]

_# Check the values and dimensions (shape) of the vectors_

print('Theta to be estimated:')

print()

print ('First 10 rows and 5 columns of X and Y:')

print()

print()

print('Dimensions of X and Y:')

print('shape', 'shape')

**Task 2: Use the normal equation**

Use the following function to solve the linear regression model, optimising its coefficients Θ, by completing the following cell:

In [ ]:

_# TODO: Use the function that solves the normal equation_

**def** normal\_equation(x, y):

"""Calculate the optimal theta using the normal equation for multivariate linear regression

Positional arguments:

x -- Numpy 2D array with the values of the independent variables from the examples, of size m x n

y -- Numpy 1D array with the dependent/target variable, of size m x 1

Return:

theta -- Numpy 1D array with the weights of the model coefficients, of size 1 x n (row vector)

"""

theta **=** [**...**]

**return** theta

**Task 3: Check the implementation**

Use the synthetic dataset you created earlier to check that your implementation returns the same or very close to the original Θ value.

Try to check it several times, changing parameters such as the number of examples and the number of features.

Also, add an error term to the Y again. In this case, the initial and final Θ will not quite match as we have introduced error or "noise" into the training dataset.

Sometimes, the normal equation is not invertible, so you may encounter that error. In that case, don't worry, it is a limitation of the analytical method and not of your particular implementation if it works in the other cases.

In [ ]:

_# TODO: Check the implementation of your normal equation_

theta **=** normal\_equation(X, Y)

print('Theta original:')

print(Theta\_verd)

print('Theta estimated:')

print(theta)

print('Difference between both Thetas:')

print(Theta\_verd **-** theta)
