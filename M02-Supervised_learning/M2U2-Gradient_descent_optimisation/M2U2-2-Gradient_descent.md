# **Multivariate Linear Regression: Gradient descent**

M2U2 - Exercise 2

**What are we going to do?**

- Implement the optimization of the cost function using gradient descent, or in other words, training the model

Remember to follow the instructions for the submission of assignments indicated in [Submission Instructions](https://github.com/Tokio-School/Machine-Learning/blob/main/Instrucciones%20entregas.md).

**Instructions**

This exercise is a continuation of the previous exercise "Cost function", so you should build on it.

In [ ]:

**import** time

**import** numpy **as** np

**from** matplotlib **import** pyplot **as** plt

**Task 1: Implement the cost function for multivariate linear regression**

In this task, you must copy the corresponding cell from the previous exercise, bringing your code to implement the vectorised cost function:

In [ ]:

_# TODO: Implement the vectorised cost function following the template below._

**def** cost\_function(x, y, theta):

""" Compute the cost function for the considered dataset and coefficients.

Positional arguments:

x -- Numpy 2D array with the values of the independent variables from the examples, of size m x n

y -- Numpy 1D array with the dependent/target variable, of size m x 1

theta -- Numpy 1D array with the weights of the model coefficients, of size 1 x n (row vector)

Return:

j -- float with the cost for this theta array

"""

m **=** [**...**]

_# Remember to check the dimensions of the matrix multiplication to do it correctly_

j **=** [**...**]

**return** j

**Task 2: Implement the optimisation of this cost function using gradient descent**

We are now going to solve the optimisation of this cost function to train the model, using the vectorised gradient descent method. The model will be considered trained when its cost function has reached a minimum, stable value.

Y=hΘ(X)=X×ΘTJθ=12m∑i=0m(hθ(xi)−yi)2θj:=θj−α[1m∑i=0m(hθ(xi)−yi)xji

To do this, once again, fill in the code template in the next cell.

Tips:

- If you prefer, you can first implement the function with loops and iterations, and finally in a vectorised way
- Remember the dimensions of each vector/matrix
- Again, record the operations in step-by-step order on a sheet or in an auxiliary cell
- At each step, write down the dimensions of your result, which you can also check in your code
- Use numpy.matmul() for matrix multiplication
- At the start of each training iteration, you must copy all Θ values, since you are going to iterate by updating each of its values based on the entire vector

In [ ]:

_# TODO: Implement the function that trains the model using gradient descent_

**def** gradient\_descent(x, y, theta, alpha, e, iter\_):

""" Trains the model by optimising its gradient descent cost function

Positional arguments:

x -- Numpy 2D array with the values of the independent variables from the examples, of size m x n

y -- Numpy 1D array with the dependent/target variable, of size m x 1

theta -- Numpy 1D array with the weights of the model coefficients, of size 1 x n (row vector)

alpha -- float, training rate

Named arguments (keyword):

e -- float, minimum difference between iterations to declare that the training has finally converged

iter\_ -- int/float, nº of iterations

Return:

j\_hist -- list/array with the evolution of the cost function during the training

theta -- NumPy array with the value of theta at the last iteration

"""

_# TODO: enters default values for e and iter\_ in the function keyword arguments_

iter\_ **=** int(iter\_) _# If you have entered iter\_ in scientific notation (1E3) or float (1000.), converts it_

_# Initialises j\_hist as a list or a NumPy array. Remember that we do not know what size it will eventually be_

_#Your max. nº of elements will be the max. nº of iterations_

j\_hist **=** [**...**]

m, n **=** [**...**] _# Obtain m and n from the dimensions of X_

**for** iter\_ **in** [**...**]: _# Iterate over the maximum nº of iterations_

theta\_iter **=** [**...**] _# Copy the theta for each iteration with "deep copy", since we have to update it_

**for** j **in** [**...**]: _# Iterate over the nº of features_

_# Update theta\_iter for each feature, according to the derivative of the cost function_

_# Include the training rate alpha_

_# Careful with the matrix multiplication, its order and dimensions_

theta\_iter[j] **=** theta[j] **-** [**...**]

theta **=** theta\_iter _# Updates the entire theta, ready for the next iteration_

cost **=** cost\_function([**...**]) _# Calculates the cost for the current iteration of theta_

j\_hist[**...**] _# Adds the cost of the current iteration to the cost history_

_# Check if the difference between the cost of the current iteration and that of the last iteration in absolute value_

_# is less than the minimum difference to declare convergence, e, for all iterations_

_# except the first_

**if** k **\>** 0 **and** [**...**]:

print('Converge at iteration nº: ', k)

**break**

**else:**

print('Max. nº of iterations reached')

**return** j\_hist, theta

**Task 3: Check the implementation of gradient descent**

To check your implementation, again, use the same cell, varying its parameters several times, plotting the evolution of the cost function and seeing how its value approaches 0.

In each case, check that the initial and final Θ are very similar in the following scenarios:

1. It generates several synthetic datasets, testing each of them
2. It modifies the nº of examples and features, m and n
3. It modifies the error parameter, which may cause the initial and final Θ to not quite match, and the greater the error, the more difference there may be
4. Check the max. nº of iterations or the training rate α hyperparameters, which will make the model take more or less time to train, within minimum and maximum values

In [ ]:

_# TODO: Generate a synthetic dataset with an error term in whatever way you choose, with NumPy or Scikit-learn._

m = 0

n = 0

e **=** 0.

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

In [ ]:

_# TODO: Test your implementation by training a model on the previously created synthetic dataset_

_# Use a randomly initiated theta or the Theta\_verd, depending on the scenario to be tested_

theta\_ini **=** [**...**]

print('Theta initial:')

print(theta\_ini)

alpha **=** 1e-1

e **=** 1e-3

iter\_ **=** 1e3 _# Check that your function can support float values or modify it._

print('Hyperparameters used:')

print('Alpha:', alpha, 'Error max.:', e, 'Nº iter', iter\_)

t **=** time **.** time()

j\_hist, theta\_final **=** gradient\_descent([**...**])

print('Training time (s):', time **.** time() **-** t)

_# TODO: complete_

print('\nLast 10 values of the cost function')

print(j\_hist[**...**])

print('\Final cost:')

print(j\_hist[**...**])

print('\nTheta final:')

print(theta\_final)

print('True values of Theta and difference with trained values:')

print(Theta\_verd)

print(theta\_final **-** Theta\_verd)

Plot the history of the cost function to check your implementation:

In [ ]:

_# TODO: Plot the cost function vs. the number of iterations._

plt **.** figure()

plt **.** title('Cost function')

plt **.** xlabel(nº iterations)

plt **.** ylabel('cost')

plt **.** plot([**...**]) _# Complete_

plt **.** grid()

plt **.** show()

To fully test the implementation of these functions, modify the original synthetic dataset to check that the cost function and gradient descent training still work correctly.

E.g., modify the nº of examples and the nº of features.

Also, add an error term to Y again. In this case, the initial and final Θ may not quite match as we have introduced error or "noise" into the training dataset.

Finally, check all the hyperparameters of your implementation. Use several values of alpha, e, nº of iterations, etc., and check that the results are as expected.
