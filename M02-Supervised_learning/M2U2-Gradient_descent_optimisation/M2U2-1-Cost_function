# **Multivariate Linear Regression: Cost function**

M2U2 - Exercise 1

**What are we going to do?**

- Implement the cost function for multivariate linear regression

Remember to follow the instructions for the submission of assignments indicated in [Submission Instructions](https://github.com/Tokio-School/Machine-Learning/blob/main/Instrucciones%20entregas.md).

In [1]:

**import** numpy **as** np

**Task 1: Implement the cost function for multivariate linear regression**

In this task, you must implement the cost function for multivariate linear regression in Python using NumPy. The cost function must follow the function included in the slides and in the course manual.

To do this, first fill in the code in the following cell to implement the cost function.

Recall the equation:

Y=hΘ(X)=X×ΘTJθ=12m∑i=0m(hθ(xi)−yi)2

To implement it, follow these steps:

1. Take some time to review the equation and make sure you understand all the mathematical operations reflected in it
2. Go back to previous exercises or review the slides and write down on a sheet of paper (or auxiliary cell) the dimensions of each vector or matrix in the equation
3. Write down the linear algebra operations step by step on this paper or in an auxiliary cell
  1. Start by substituting hθ in the 2nd equation for its value from the 1st equation
  2. The first operation is to find the predicted hθ or Y for each row of X (multiplying it by Θ)
  3. 2nd, subtract the value of Y for this example/row of X, finding its residual
  4. Then square the result
  5. Then, sum all the squares of the residuals for all examples/rows of X
  6. Finally, divide them by 2 \* m
4. Write down next to each step the dimensions that your result should have. Remember that the final result of the cost function is a scalar or number
5. Finally, think about how to iterate with for loops for each value of X, Θ, and Y, to implement the cost function:
  1. Iterate over all the rows or examples of X (m rows)
  2. Within that loop, iterate over the features or values of X and Θ to calculate the predicted Y for that example
  3. Once all the residuals have been found, find the total cost

_Notes:_

- The steps mentioned above are only a guide, an assist. In each exercise, implement your code in your own way, with the approach you prefer, using the cell code scheme or not
- Don't worry too much for now about whether it is working correctly or not, as we will check it in the next task. If there are any errors, you can return to this cell to correct your code.

In [ ]:

_# TODO: Implement the non-vectorised cost function following the template below_

**def** cost\_function\_non\_vectorised(x, y, theta):

""" Computes the cost function for the considered dataset and coefficients.

Positional arguments:

x -- Numpy 2D array with the values of the independent variables from the examples, of size m x n

y -- Numpy 1D array with the dependent/target variable, of size m x 1

theta -- Numpy 1D array with the weights of the model coefficients, of size 1 x n (row vector)

Return:

j -- float with the cost for this theta array

"""

m **=** [**...**]

_# Remember to check the dimensions of the matrix multiplication to do it correctly._

j **=** [**...**]

**return** j

**Task 2: Check your implementation**

To test your implementation, retrieve your code from the previous notebook about synthetic datasets for multivariate linear regression and use it to generate a dataset in the following cell:

In [ ]:

_# TODO: Generate a synthetic dataset, with error term, in the form of your choice, with NumPy or Scikit-learn_

m = 0

n = 0

e **=** 0

X **=** [**...**]

Theta\_verd **=** [**...**]

Y **=** [**...**]

_# Check the values and dimensions (form or "shape") of the vectors_

print('Actual theta to be estimated:')

print()

print('First 10 rows and 5 columns of X and Y:')

print()

print()

print('Dimensions of X and Y:')

print('shape', 'shape')

Now let's check your implementation of the cost function in the following cells.

Remember that the cost function represents the "error" of your model, the sum of the squares of the residuals of your model.

Therefore, the cost function has the following features:

- It has no units, so we cannot know if its value is "too high or too low", just compare the costs of two different models (sets of Θ)
- It has a value of 0 for the theoretically optimal Θ
- Its values are always positive
- It has a higher value the further away the Θ used is from the optimal Θ
- Its value increases with the square of the residuals of the model

Therefore, use the next cell to check the implementation of your function with different Θ's, correcting your function if necessary. Check that:

1. If Θ is equal to the Θverd (obtained when defining the dataset), the cost is 0
2. If Θ is different than Θverd, the cost is non-0 and positive.
3. The further away Θ is from the Θverd, the higher the cost (check this with 3 different Θ's other than Θverd, in order from lowest to highest)

_Note:_ For this, use the same cell, modifying its variables several times.

In [ ]:

_# TODO: Check the implementation of your cost function_

theta **=** Theta\_verd _# Modifiy and test several values of theta_

j **=** cost\_function\_non\_vectorized(X, Y, theta)

print('Cost of the model:')

print(j)

print('Theta checked and actual Theta:')

print(theta)

print(Theta\_verd)

**Task 3: Vectorise the cost function**

Now we are going to implement a new cost function, but this time vectorised.

A vectorised function is one that is implemented based on linear algebraic operations, instead of e.g., the for loops used in the first function, and therefore its computation is much faster and more efficient, especially if it is run on GPUs or specialised processors.

It implements the cost function again, but this time using exclusively linear algebra operations to operate with NumPy vectors/arrays.

Tips:

- Check the dimensions of the result of each operation or intermediate step, one by one if necessary
- Try to implement the equation with as few operations as possible, without loops or iterations.
- Use functions like [numpy.matmul()](https://numpy.org/doc/stable/reference/generated/numpy.matmul.html) or numpy.sum().

In [ ]:

_# TODO: Implement the non-vectorised cost function following the template below_

**def** cost\_function(x, y, theta):

""" Computes the cost function for the considered dataset and coefficients.

Positional arguments:

x -- Numpy 2D array with the values of the independent variables from the examples, of size m x n

y -- Numpy 1D array with the dependent/target variable, of size m x 1

theta -- Numpy 1D array with the weights of the model coefficients, of size 1 x n (row vector)

Return:

j -- float with the cost for this theta array

"""

m **=** [**...**]

_# Remember to check the dimensions of the matrix multiplication to do it correctly._

j **=** [**...**]

**return** j

Finally, go back to task 2 and repeat the same steps to now check your vectorized function.
