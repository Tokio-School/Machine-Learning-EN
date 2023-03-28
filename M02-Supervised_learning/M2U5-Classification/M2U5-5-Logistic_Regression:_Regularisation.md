# **Logistic Regression: Regularisation**

M2U5 - Exercise 5

**What are we going to do?**

- We will implement the regularised cost and gradient descent functions
- We will check the training by plotting the evolution of the cost function
- We will find the optimal _lambda_ regularisation parameter using validation

Remember to follow the instructions for the submission of assignments indicated in [Submission Instructions](https://github.com/Tokio-School/Machine-Learning/blob/main/Instrucciones%20entregas.md).

**Instructions**

Once the unregularised cost function and gradient descent are implemented, we will regularise them and train a full logistic regression model, checking it by validation and evaluating it on a test subset.

In [ ]:

**import** time

**import** numpy **as** np

**from** matplotlib **import** pyplot **as** plt

**Create a synthetic dataset for logistic regression**

We will create a synthetic dataset with only 2 classes (0 and 1) to test this implementation of a fully trained binary classification model, step by step.

To do this, manually create a synthetic dataset for logistic regression with bias and error term (to have _Theta\_true_ available) with the code you used in the previous exercise:

In [ ]:

_# TODO: Manually generate a synthetic dataset with a bias term and an error term_

m = 100

n = 1

_# Generate a 2D m x n array with random values between -1 and 1_

_# Insert a bias term as a first column of 1's_

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

**Implement the sigmoid activation function**

Copy your cell with the sigmoid function:

In [ ]:

_# TODO: Implement the sigmoid function_

**Preprocess the data**

As we did for linear regression, we will preprocess the data completely, following the usual 3 steps:

- Randomly reorder the data.
- Normalise the data.
- Divide the dataset into training, validation, and test subsets.

You can do this manually or with Scikit-learn's auxiliary functions.

**Randomly rearrange the dataset**

Reorder the data in the _X_ and _Y_ dataset:

In [ ]:

_# TODO: Randomly reorder the dataset_

print('First 10 rows and 5 columns of X and Y:')

print()

print()

print('Reorder X and Y:')

_# Use an initial random state of 42, in order to maintain reproducibility_

X, Y **=** [**...**]

print('First 10 rows and 5 columns of X and Y:')

print()

print()

print('Dimensions of X and Y:')

print()

**Normalise the dataset**

Implement the normalisation function and normalise the dataset of _X_ examples:

In [ ]:

_# TODO: Normalise the dataset with a normalisation function_

_# Copy the normalisation function you used in the linear regression exercise_

**def** normalize(x, mu, std):

**pass**

_# Find the mean and standard deviation of the features of X (columns), except the first column (bias)_

mu **=** [**...**]

std **=** [**...**]

print('Original X:')

print(X)

print(X **.** shape)

print('Mean and standard deviation of the features:')

print()

print(mu **.** shape)

print(std)

print(std **.** shape)

print('Normalized X:')

X\_norm **=** np **.** copy(X)

X\_norm[...] = normalize(X[...], mu, std) _# Normalize only column 1 and the subsequent columns, not column 0_

print(X\_norm)

print(X\_norm **.** shape)

_Note:_ If you had modified your _normalize_ function to calculate and return the values of _mu_ and _std_, you can modify this cell to include your custom code.

**Divide the dataset into training, validation, and test subsets**

Divide the _X_ and _Y_ dataset into 3 subsets with the usual ratio, 60%/20%/20%.

If your number of examples is much higher or lower, you can always modify this ratio to another ratio such as 50/25/25 or 80/10/10.

In [ ]:

_# TODO: # Divide the X and Y dataset into the 3 subsets following the indicated ratios_

ratio **=** [60, 20, 20]

print('Ratio:\n', ratio, ratio[0] **+** ratio[1] **+** ratio[2])

r **=** [0, 0]

_# Tip: the round() function and the x.shape attribute may be useful to you_

r[0] **=** [**...**]

r[1] **=** [**...**]

print('Cutoff indices:\n', r)

_# Tip: the np.array\_split() function may be useful to you_

X\_train, X\_val, X\_test **=** [**...**]

Y\_train, Y\_val, Y\_test **=** [**...**]

print('Size of the subsets:')

print(X\_train **.** shape)

print(Y\_train **.** shape)

print(X\_val **.** shape)

print(Y\_val **.** shape)

print(X\_test **.** shape)

print(Y\_test **.** shape)

**Implement the sigmoid activation function**

Copy your cell with the sigmoid function:

In [ ]:

_# TODO: Implement the sigmoid function_

**Implement the regularised cost function**

We are going to implement the regularised cost function. This function will be similar to the one we implemented for linear regression in a previous exercise.

Regularised cost function

Y=hΘ(x)=g(X×ΘT)J(Θ)=−[1m∑i=0m(yilog(hθ(xi))+(1−yi)log(1−hθ(xi))]+λ2m∑j=1nΘj2

In [ ]:

_# TODO: Implement the regularised cost function for logistic regression_

**def** regularized\_logistic\_cost\_function(x, y, theta, lambda\_ **=** 0.):

""" Computes the cost function for the considered dataset and coefficients

Positional arguments:

x -- 2D ndarray with the values of the independent variables from the examples, of size m x n

y -- 1D ndarray with the dependent/target variable, of size m x 1 and values of 0 or 1

theta -- 1D ndarray with the weights of the model coefficients, of size 1 x n (row vector)

lambda\_ -- regularisation factor, by default 0.

Return:

j -- float with the cost for this theta array

"""

m **=** [**...**]

_# Remember to check the dimensions of the matrix multiplication to perform it correctly_

j **=** [**...**]

_# Regularise for all Theta except the bias term (the first value)_

j **+=** [**...**]

**return** j

Now let's check your implementation in the following scenarios:

1. For _lambda_ = 0.
  1. Using _Theta\_true_, the cost should be 0.
  2. As the value of _theta_ moves away from _Theta\_true_, the cost should increase.
2. For _lambda_ ! = 0:
  1. Using _Theta\_true_, the cost should be greater than 0.
  2. The higher the _lambda_, the higher the cost.
  3. The increase in cost as a function of _lambda_ is exponential.

In [ ]:

_# TODO: Test your implementation on the dataset_

theta **=** Theta\_true _# Modify and test several values of theta_

j **=** logistic\_cost\_function(X, Y, theta)

print('Cost of the model:')

print(j)

print('Checked theta and Actual theta:')

print(theta)

print(Theta\_true)

Record your experiments and results in this cell (in Markdown or code):

1. Experiment 1
2. Experiment 2
3. Experiment 3
4. Experiment 4
5. Experiment 5

**Train an initial model on the training subset**

As we did in previous exercises, we will train an initial model to check that our implementation and the dataset work correctly, and then we will be able to train a model with validation without any problem.

To do this, follow the same steps as you did for linear regression:

- Train an initial model without regularisation.
- Plot the history of the cost function to check its evolution.
- If necessary, modify any of the parameters and retrain the model. You will use these parameters in the following steps.

Copy the cells from previous exercises where you implemented the cost function in unregularised logistic regression and the cell where you trained the model, and modify them for regularised logistic regression.

Recall the gradient descent functions for regularised logistic regression:

Y=hΘ(x)=g(X×ΘT)θ0:=θ0−α1m∑i=0m(hθ(xi)−yi)x0iθj:=θj−α[1m∑i=0m(hθ(xi)−yi)xji+λmθj]; j∈[1,n]θj:=θj(1−αλm)−α1m∑i=0m(hθ(xi)−yi)xji; j∈[1,n]

In [ ]:

_# TODO: Copy the cell with the gradient descent for unregularised logistic regression and modify it to implement the regularisation_

In [ ]:

_# TODO: Copy the cell where we trained the model_

_# Train your model on the unregularised training subset and check that it works correctly_

In [ ]:

_# TODO: Plot the evolution of the cost function vs. the number of iterations_

plt **.** figure(1)

**Check the implementation**

Check your implementation again, as you did in the previous exercise.

On this occasion, it also shows that for a _lambda_ other than 0, the higher the _lambda_ the higher the cost will be, due to the penalty.

**Check for deviation or overfitting**

As we did in linear regression, we will check for overfitting by comparing the cost of the model on the training and validation datasets:

In [ ]:

_# TODO: Check the cost of the model on the training and validation datasets._

_# Use the Theta\_final of the trained model in both cases_

Remember that with a random synthetic dataset it is difficult to have one or the other, but by proceeding in this way we will be able to identify the following problems:

- If the final cost in both subsets is high, there may be a problem with deviation or bias.
- If the final costs in both subsets are very different from each other, there may be a problem with overfitting orvariance_._

**Find the optimal** _ **lambda** _ **hyperparameter using validation**

As we have done in previous exercises, we will optimise our regularisation parameter by validation.

To do this, we will train a different model on the training subset for each _lambda_ value to be considered, and evaluate its error or final cost on the validation subset.

We will plot the error of each model vs. the _lambda_ value used and implement a code that will automatically choose the most optimal model among all of them.

Remember to train all your models under equal conditions:

In [ ]:

_# TODO: Train a model on X\_train for each different lambda value and evaluate it on X\_val_

_# Use a logarithmic space between 10 and 10^3 with 10 elements with non-zero decimal values starting with a 1 or a 3_

lambdas **=** [**...**]

_# Complete the code to train a different model for each class and value of lambda on X\_train_

_# Store your theta and final cost/error_

_# Afterwards, evaluate its total cost on the validation subset_

_# Store this information in the following arrays, which are the same size as lambda's arrays_

j\_train **=** [**...**]

j\_val **=** [**...**]

theta\_val **=** [**...**]

In [ ]:

_# TODO: Plot the final error for each value of lambda_

plt **.** figure(2)

_# Fill in your code_

**Choosing the best model**

Copy the code from previous exercises and modify it to choose the most accurate model on the validation subset for each class:

In [ ]:

_# TODO: Choose the optimal model and lambda value, with the lowest error on the CV subset_

_# Iterate over all the combinations of theta and lambda and choose the one with the lowest cost on the CV subset_

j\_final **=** [**...**]

theta\_final **=** [**...**]

lambda\_final **=** [**...**]

**Evaluate the model on the test subset**

Finally, we will evaluate the model on a subset of data that we have not used for its training nor to choose any of its hyperparameters.

Therefore, we will calculate the total error or cost on the test subset and graphically check the residuals of the model on it:

In [ ]:

_# TODO: Calculate the model error on the test subset using the cost function with the corresponding theta and lambda_

j\_test **=** [**...**]

In [ ]:

_# TODO: Calculate the predictions of the model on the test subset, calculate the residuals and plot them against the index of examples (m)_

_# Remember to use the sigmoid function to transform the predictions_

Y\_test\_pred **=** [**...**]

residuals **=** [**...**]

plt **.** figure(3)

_# Fill in your code_

plt **.** show()

**Make predictions about new examples**

With our model trained, optimised, and evaluated, all that remains is to put it to work by making predictions with new examples.

To do this, we will:

- Generate a new example, which follows the same pattern as the original dataset.
- Normalise its features before making predictions about them.
- Generate a prediction for this new example.

In [ ]:

_# TODO: Generate a new example following the original pattern, with a bias term and a random error term._

X\_pred **=** [**...**]

_# Normalise its features (except the bias term) with the original means and standard deviations_

X\_pred **=** [**...**]

_# Generate a prediction for this new example_

Y\_pred **=** [**...**]
