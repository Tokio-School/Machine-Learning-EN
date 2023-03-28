# **Linear Regression: Regularisation**

M2U3 - Exercise 2

**What are we going to do?**

- We will implement a regularised cost function for multivariate linear regression
- We will implement the regularisation for gradient descent

Remember to follow the instructions for the submission of assignments indicated in [Submission Instructions](https://github.com/Tokio-School/Machine-Learning/blob/main/Instrucciones%20entregas.md).

In [ ]:

**import** time

**import** numpy **as** np

**from** matplotlib **import** pyplot **as** plt

**Creation of a synthetic dataset**

To test your implementation of a regularised gradient descent and cost function, retrieve your cells from the previous notebooks on synthetic datasets and generate a dataset for this exercise.

Don't forget to add a bias term to _X_ and an error term to _Y_, initialized to 0 for now.

In [ ]:

_# TODO: Manually generate a synthetic dataset, with a bias term and an error term initialised to 0_

m = 1000

n = 3

X **=** [**...**]

Theta\_true **=** [**...**]

error **=** 0.

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

**Regularised cost function**

We will now modify our implementation of the cost function from the previous exercise to add the regularisation term.

Recall that the regularised cost function is:

hθ(xi)=Y=X×ΘTJθ=12m[∑i=0m(hθ(xi)−yi)2+λ∑j=1nθj2]

In [ ]:

_# TODO: Implement the regularised cost function according to the following template_

**def** regularized\_cost\_function(x, y, theta, lambda\_ **=** 0.):

""" Computes the cost function for the considered dataset and coefficients.

Positional arguments:

x -- Numpy 2D array with the values of the independent variables from the examples, of size m x n

y -- Numpy 1D array with the dependent/target variable, of size m x 1

theta -- Numpy 1D array with the weights of the model coefficients, of size 1 x n (row vector)

Named arguments:

lambda -- float with the regularisation parameter

Return:

j -- float with the cost for this theta array

"""

m **=** [**...**]

_# Remember to check the dimensions of the matrix multiplication to perform it correctly_

_# Remember not to regularize the coefficient of the bias parameter (first value of theta)._

j **=** [**...**]

**return** j

_NOTE:_ Check that the function simply returns a float value, and not an array or matrix. Use the ndarray.resize((size0, size1) )method if you need to change the dimensions of any array before you multiply it with np.matmul() and make sure the result dimensions match, or returns j[0,0] as the float value.

As the synthetic dataset has the error term set at 0, the result of the cost function for the _Theta\_true_ with parameter _lambda_ = 0 must be exactly 0.

As before, as we move away with different values of θ, the cost should increase. Similarly, the higher the _lambda_ regularisation parameter, the higher the penalty and cost, and the higher the _Theta_ value, the higher the penalty and cost as well.

Check your implementation in these 5 scenarios:

1. Using _Theta\_true_ and with _lambda_ at 0, the cost should still be 0.
2. With _lambda_ still at 0, as the value of _theta_ moves away from _Theta\_true_, the cost should increase.
3. Using _Theta\_true_ and with a _lambda_ other than 0, the cost must now be greater than 0.
4. With a _lambda_ other than 0, for a _theta_ other than _Theta\_true,_ the cost must be higher than with _lambda_ equal to 0.
5. With a _lambda_ other than 0, the higher the values of the coefficients of _theta_ (positive or negative), the higher the penalty and the higher the cost.

Recall that the value of _lambda_ must always be positive and generally less than 0 [0, 1e-1, 3e-1, 1e-2, 3e-2, ...]

In [ ]:

_# TODO: Check the implementation of your regularised cost function in these scenarios_

theta **=** Theta\_true _# Modify and test various values of theta_

j **=** regularized\_cost\_function(X, Y, theta)

print('Cost of the model:')

print(j)

print('Tested Theta and actual Theta:')

print(theta)

print(Theta\_true)

Record your experiments and results in this cell (in Markdown or code):

1. Experiment 1
2. Experiment 2
3. Experiment 3
4. Experiment 4
5. Experiment 5

**Regularised gradient descent**

Now we will also regularise the training by gradient descent. We will modify the _Theta_ updates so that they now also contain the _lambda_ regularisation parameter:

θ0:=θ0−α1m∑i=0m(hθ(xi)−yi)x0iθj:=θj−α[1m∑i=0m(hθ(xi)−yi)xji+λmθj]; j∈[1,n]θj:=θj(1−αλm)−α1m∑i=0m(hθ(xi)−yi)xji; j∈[1,n]

Remember to build again on your previous implementation of the gradient descent function.

In [ ]:

_# TODO: Implement the function that trains the regularised gradient descent model_

**def** regularized\_gradient\_descent(x, y, theta, alpha, lambda\_ **=** 0., e, iter\_):

""" Trains the model by optimising its cost function using gradient descent

Positional arguments:

x -- Numpy 2D array with the values of the independent variables from the examples, of size m x n

y -- Numpy 1D array with the dependent/target variable, of size m x 1

theta -- Numpy 1D array with the weights of the model coefficients, of size 1 x n (row vector)

alpha -- float, training rate

Named arguments (keyword):

lambda -- float with the regularisation parameter

e -- float, minimum difference between iterations to declare that the training has finally converged

iter\_ -- int/float, nº of iterations

Return:

j\_hist -- list/array with the evolution of the cost function during the training

theta -- Numpy array with the value of theta at the last iteration

"""

_# TODO: enters default values for e and iter\_ in the function keyword arguments_

iter\_ **=** int(iter\_) _# If you have entered iter\_ in scientific notation (1e3) or float (1000.), converts it_

_# Initialises j\_hist as a list or a Numpy array. Remember that we do not know what size it will eventually be_

j\_hist **=** [**...**]

m, n **=** [**...**] _# Obtain m and n from the dimensions of X_

**for** k **in** [**...**]: _# Iterate over the maximum nº of iterations_

_# Declare a theta for each iteration as a "deep copy" of theta, since we must update it value by value_

theta\_iter **=** [**...**]

**for** j **in** [**...**]: _# Iterate over the nº of features_

_# Update theta\_iter for each feature, according to the derivative of the cost function_

_# Include the training rate alpha_

_# Careful with the matrix multiplication, its order and dimensions_

**if** j **\>** 0:

_# Regularise all coefficients except for the bias parameter (first coef.)_

**pass**

theta\_iter[j] **=** theta[j] **-** [**...**]

theta **=** theta\_iter

cost **=** cost\_function([**...**]) _# Calculates the cost for the current theta iteration_

j\_hist[**...**] _# Adds the cost of the current iteration to the cost history._

_# Check if the difference between the cost of the current iteration and that of the last iteration in absolute value_

_is less than the minimum difference to declare convergence, e_

**if** k **\>** 0 **and** [**...**]:

print('Converge at iteration nº: ', k)

**break**

**else:**

print('Max. nº of iterations reached')

**return** j\_hist, theta

_Note:_ Remember that the code templates are only an aid. Sometimes, you may want to use different code with the same functionality, e.g., iterate over elements in a different way, etc. Feel free to modify them as you wish!

**Checking the regularised gradient descent**

To check your implementation again, check with _lambda_ at 0 using various values of _theta\_ini_, both with the _Theta\_true_ and values further and further away from it, and check that eventually the model converges to the _Theta\_true_:

In [ ]:

_# TODO: Test your implementation by training a model on the previously created synthetic dataset_

_# Create an initial theta with a given, random, or hand-picked value_

theta\_ini **=** [**...**]

print('Theta initial:')

print(theta\_ini)

alpha **=** 1e-1

lambda\_ **=** 0.

e **=** 1e-3

iter\_ **=** 1e3 _# Check that your function supports float values or modify it_

print('Hyperparameters used:')

print('Alpha:', alpha, 'Error máx.:', e, 'Nº iter', iter\_)

t **=** time **.** time()

j\_hist, theta\_final **=** regularized\_gradient\_descent([**...**])

print('Training time (s):', time **.** time() **-** t)

_# TODO: complete_

print('\nLast 10 cost function values')

print(j\_hist[**...**])

print('\Final cost:')

print(j\_hist[**...**])

print('\nTheta final:')

print(theta\_final)

print(True values of Theta and difference with trained values:')

print(Theta\_true)

print(theta\_final **-** Theta\_true)

Now recheck the training of a model in some of the preceding scenarios:

1. Using a random _theta\_ini_ and with _lambda_ at 0, the final cost should still be close to 0 and the final _theta_ close to _Theta\_true_.
2. Using a random _theta\_ini_ and with a small non-zero _lambda_, the final cost should be close to 0, although the accuracy of the model may decrease.
3. As the _lambda_ value increases, the accuracy of the model will decrease.

To do this, remember that you can modify the values of the cells and re-execute them.

Record your experiments and results in this cell (in Markdown or code):

1. Experiment 1
2. Experiment 2
3. Experiment 3
4. Experiment 4
5. Experiment 5

**Why did we need to use regularisation?**

The aim of regularisation was to penalize the model when it suffers from overfitting, when the model starts to memorise results rather than learning to generalise.

This is a problem when the training data and the data on which we must make predictions in production follow significantly different distributions.

To test our training with regularised gradient descent, go back to the dataset generation section and generate a dataset with a much lower ratio of examples to features and a much higher error rate.

Start playing with these values and then modify the _lambda_ of the model to see if a _lambda_ value other than 0 starts to be more accurate than _lambda_ = 0.
