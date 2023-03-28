# **Linear Regression: Synthetic dataset example**

M2U2 - Exercise 5

**What are we going to do?**

- Use an automatically generated synthetic dataset to check our implementation
- Train a multivariate linear regression ML model
- Check the training evolution of the model
- Evaluate a simple model
- Make predictions about new future examples

Remember to follow the instructions for the submission of assignments indicated in [Submission Instructions](https://github.com/Tokio-School/Machine-Learning/blob/main/Instrucciones%20entregas.md).

In [ ]:

**import** time

**import** numpy **as** np

**from** matplotlib **import** pyplot **as** plt

**Creation of a synthetic dataset**

We are going to create a synthetic dataset to check our implementation.

Following the methods that we have used in previous exercises, create a synthetic dataset using the NumPy method.

Include a controllable error term in that dataset, but initialise it to 0, since to make the first implementation of this multivariate linear regression ML model we do not want any error in the data that could hide an error in our model.

Afterwards, we will introduce an error term to check that our implementation can also train the model under these more realistic circumstances.

**The bias or intercept term**

This time, we are going to generate the synthetic dataset with a small modification: we are going to add a first column of 1's to X, or a 1. (float) as the first value of the features of each example.

Furthermore, since we have added one more feature n to the matrix X, we have also added one more feature or value to the vector Θ, so we now have n + 1 features.

Why do we add this column, this new term or feature?

Because this is the simplest way to implement the linear equation in a single linear algebra operation, i.e., to vectorise it.

In this way, we thus convert Y=m×X+b into Y=X×Θ, saving us an addition operation and implementing the equation in a single matrix multiplication operation.

The term _b_, therefore, is incorporated as the first term of the vector Θ, which when multiplied by the first column of X, which has a value of 1 for all its rows, allows us to add said term _b_ to each example.

In [ ]:

_# TODO: Generate a synthetic dataset in whatever way you choose, with error term initially set to 0._

m = 100

n = 3

_# Create a matrix of random numbers in the interval [-1, 1)_

X **=** [**...**]

_# Insert a vector of 1's as the 1__st_ _column of X_

_# Tips:__np.insert(), np.ones(), index 0, axis 1..._

X **=** [**...**]

_Generate a vector of random numbers in the interval [0, 1) of size n + 1 (to add the bias term)_

Theta\_verd **=** [**...**]

_# Add to the Y vector a random error term in % (0.1= 10%) initialised at 0_

_Said term represents an error of +/- said percentage, e.g., +/- 5%,+/- 10%, etc., not just to add_

_# The percentage error is calculated on Y, therefore the error would be e.g., +3.14% of Y, or -4.12% of Y...._

error **=** 0.

Y **=** np **.** matmul(X, Theta\_verd)

Y **=** Y **+** [**...**] **\*** error

_# Check the values and dimensions of the vectors_

print('Theta to be estimated and its dimensions:')

print()

print()

print('First 10 rows and 5 columns of X and Y:')

print()

print()

print('Dimensions of X and Y:')

print()

Note the matrix multiplication operation implemented: Y=X×Θ

Check the dimensions of each vector: X, Y, Θ. _Do you think this operation is possible according to the rules of linear algebra?_

If you have doubts, you can consult the NumPy documentation relating to the np.matmul function.

Check the result, perhaps reducing the original number of examples and features, and make sure it is correct.

**Training the model**

Copy your implementation of the cost function and its optimisation by gradient descent from the previous exercise:

In [ ]:

_# TODO: Copy the code of your cost and gradient descent functions_

**def** cost\_function(x, y, theta):

""" Computes the cost function for the considered dataset and coefficients.

Positional arguments:

x -- Numpy 2D array with the values of the independent variables from the examples, of size m x n +1

y -- Numpy 1D array with the dependent/target variable, of size m x 1

theta -- Numpy 1D array with the weights of the model coefficients, of size 1 x n +1 (row vector)

Return:

j -- float with the cost for this theta array

"""

**pass**

**def** gradient\_descent(x, y, theta, alpha, e, iter\_):

""" Train the model by optimising its cost function by gradient descent

Positional arguments:

x -- Numpy 2D array with the values of the independent variables from the examples, of size m x n +1

y -- Numpy 1D array with the dependent/target variable, of size m x 1

theta -- Numpy 1D array with the weights of the model coefficients, of size 1 x n +1 (row vector)

alpha -- float, training rate

Named arguments (keyword):

e -- float, minimum difference between iterations to declare that the training has finally converged

iter\_ -- int/float, nº of iterations

Return:

j\_hist -- list/array with the evolution of the cost function during training, of size nº of iterations that the model has used

theta -- NumPy array with the value of theta at the last iteration, of size 1 x n + 1

"""

**pass**

We will use these functions to train our ML model.

Let's remind you of the steps we will follow:

- Start Θ with random values
- Optimise Θ by reducing the cost associated with each iteration of its values
- When we have found the minimum value of the cost function, take its associated Θ as the coefficients of our model

To do this, fill in the code in the following cell:

In [ ]:

_# TODO: Train your ML model by optimising its Theta coefficients using gradient descent_

_# Initialise theta with n + 1 random values_

theta\_ini **=** [**...**]

print('Theta initial:')

print(theta\_ini)

alpha **=** 1e-1

e **=** 1e-4

iter\_ **=** 1e5

print('Hyperparameters to be used:')

print('Alpha: {}, e: {}, nº max. iter: {}' **.** format(alpha, e, iter\_))

t **=** time **.** time()

j\_hist, theta **=** gradient\_descent([**...**])

print('Training time (s):', time **.** time() **-** t)

_# TODO: complete_

print('\nLast 10 values of the cost function')

print(j\_hist[**...**])

print('\Final cost:')

print(j\_hist[**...**])

print('\nTheta final:')

print(theta)

print('True values of Theta and difference with trained values:')

print(Theta\_verd)

print(theta **-** Theta\_verd)

Check that the initial Θ has not been modified. Your implementation must copy a new Python object at each iteration and not modify it during the training.

In [ ]:

_# TODO: Check that the initial Theta has not been modified._

print('Theta initial and theta final:')

print(theta\_ini)

print(theta)

**Check the training of the model**

To check the training of the model, we will graphically represent the evolution of the cost function, to ensure that there has not been any great jump and that it has been steadily moving towards a minimum value:

In [ ]:

_# TODO: Plot the evolution of the cost function vs. the number of iterations._

plt **.** figure(1)

plt **.** title('Cost function')

plt **.** xlabel('Iterations')

plt **.** ylabel('Cost')

plt **.** plot([**...**]) _# Complete the arguments_

plt **.** show()

**Making predictions**

We will use Θ, the result of our model training process, to make predictions about new examples to come in the future.

We will generate a new dataset X following the same steps that we followed previously. Therefore, if X has the same number of features (n + 1) and its values are in the same range as the previously generated X, they will behave the same as the data used to train the model.

In [ ]:

_# TODO: Make predictions using the calculated theta_

_# Generate a new matrix X with new examples. Use the same nº of features and the same range of random values,_

_but with fewer examples (e.g., 25% of the original number)_

_# Remember to add the bias term, or a first column of 1's to the matrix, of size m x n + 1._

X\_pred **=** [**...**]

_# Calculate the predictions for this new data_

y\_pred **=** [**...**] _# Hint: matmul, again_

print('Predictions:')

print(y\_pred) _# You can print the whole vector or only the first few values, if the vector is too long_

**Evaluation of the model**

We have several options for evaluating the model. At this point, we will make a simpler, quicker, and more informal assessment of the model. In subsequent modules of the course, we will look at how to evaluate our models in a more formal and precise way.

We are going to do a graphical evaluation, simply to check that our implementation works as expected:

In [ ]:

_# TODO: Plot the residuals between the initial Y and the predicted Y for the same examples_

_# Make predictions for each value of the original X with the theta trained by the model_

Y\_pred **=** [**...**]

plt **.** figure(2)

plt **.** title('Original dataset and predictions')

plt **.** xlabel('X')

plt **.** ylabel('Residuals')

_# Calculate the residuals for each example_

_# Recall that the residuals are the difference in absolute value between the actual Y and the predicted Y for each example_

residuals **=** [**...**]

_# Use a dot plot with different colours for the initial Y and the predicted Y_

plt **.** scatter([**...**])

plt **.** show()

If our implementation is correct, our model should have been trained correctly and have near-zero residuals, a near-zero difference between the original results (Y) and the results predicted by our model.

However, as we recall, in the first point we created a dataset with the error term set to 0. Therefore, each value of Y has no difference or random variation from its actual value.

In real life, either because we have not taken into account all the features that would affect our target variable, or because the data contains some small error, or because, in general, the data does not follow a completely precise pattern, we will always have some more or less random error term.

So, _what if you go back to the first cell and modify your error term, and run the steps again to train and evaluate a new linear regression model on more realistic data?_

That way you can check the robustness of your implementation.
