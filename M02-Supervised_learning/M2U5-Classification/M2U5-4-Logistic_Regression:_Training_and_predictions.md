# **Logistic Regression: Training and predictions**

M2U5 - Exercise 4

**What are we going to do?**

- We will create a synthetic dataset for logistic regression
- We will preprocess the data
- We will train the model using gradient descent
- We will check the training by plotting the evolution of the cost function
- We will make predictions about new examples

Remember to follow the instructions for the submission of assignments indicated in [Submission Instructions](https://github.com/Tokio-School/Machine-Learning/blob/main/Instrucciones%20entregas.md).

**Instructions**

Once the cost function is implemented, we will train a gradient descent logistic regression model, testing our training, evaluating it on a test subset and finally, making predictions on it.

This time we will work with a binary logistic regression, while in other exercises we will consider a multiclass classification.

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
- Divide the dataset into training and test subsets.

You can do this manually or with Scikit-learn's auxiliary functions.

**Randomly reorder the dataset**

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

**def** normalise(x, mu, std):

**pass**

_# Find the mean and standard deviation of the X features (columns), except for the first one (bias)._

mu **=** [**...**]

std **=** [**...**]

print('Original X:')

print(X)

print(X **.** shape)

print('Mean and standard deviation of the features:')

print(mu)

print(mu **.** shape)

print(std)

print(std **.** shape)

print('Normalised X:')

X\_norm **=** np **.** copy(X)

X\_norm[...] = normalize(X[...], mu, std) _# Normalise only column 1 and the subsequent columns, not column 0_

print(X\_norm)

print(X\_norm **.** shape)

**Divide the dataset into training and test subsets**

Divide the _X_ and _Y_ dataset into 2 subsets with the usual ratio of 70%/30%

If your number of examples is much higher or lower, you can always modify this ratio accordingly.

In [ ]:

_# TODO: Divide the X and Y dataset into the 2 subsets according to the indicated ratio_

ratio **=** [70, 30]

print('Ratio:\n', ratio, ratio[0] **+** ratio[1])

_# Cutoff index_

_# Tip: the round() function and the x.shape attribute may be useful to you_

r **=** [**...**]

print('Cutoff indices:\n', r)

_# Tip: the np.array\_split() function may be useful to you_

X\_train, X\_test **=** [**...**]

Y\_train, Y\_test **=** [**...**]

print('Size of the subsets:')

print(X\_train **.** shape)

print(Y\_train **.** shape)

print(X\_test **.** shape)

print(Y\_test **.** shape)

**Train an initial model on the training subset**

As we did in previous exercises, we will train an initial model to check that our implementation and the dataset work correctly, and then we will be able to train a model with validation without any problem.

To do this, follow the same steps as you did for linear regression:

- Train an initial model without implementing regularisation.
- Plot the history of the cost function to check its evolution.
- If necessary, modify any of the parameters and retrain the model. You will use these parameters in the following steps.

Copy the cells from previous exercises where you implemented the cost function for logistic regression, the unregularised gradient descent for linear regression, and the cell where you trained the regression model, and modify them for logistic regression.

Recall the gradient descent functions for logistic regression:

Y=hΘ(x)=g(X×ΘT)θj:=θj−α[1m∑i=0m(hθ(xi)−yi)xji]

In [ ]:

_# TODO: Copy the cell with the cost function_

In [ ]:

_# TODO: Copy the cell with the unregularised gradient descent function for linear regression and adapt it for logistic regression_

In [ ]:

_# TODO: Copy the cell where we trained the model_

_# Train your model on the unregularised training subset_

In [ ]:

_# TODO: Plot the evolution of the cost function vs. the number of iterations_

plt **.** figure(1)

Check your implementation in the following scenarios:

1. Using _Theta\_true_, the final cost should be practically 0 and converge in a couple of iterations.
2. As the value of _theta_ moves away from _Theta\_true_, it should need more iterations to converge, and _theta\_final_ should be very similar to _Theta\_true_.

To do this, remember that you can modify the values of the cells and re-execute them

Record your experiments and results in this cell (in Markdown or code):

1. Experiment 1
2. Experiment 2

**Evaluate the model on the test subset**

Finally, we will evaluate the model on a subset of data that we have not used to train it.

Therefore, we will calculate the total cost or error on the test subset and graphically check the residuals of the model on it:

In [ ]:

_# TODO: Calculate the error of the model on the test subset using the cost function with the corresponding theta_

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

- Generate a new example, following the same pattern as the original dataset.
- Normalise its features before making predictions about them.
- Generate a prediction for this new example.

In [ ]:

_# TODO: Generate a new example following the original pattern, with a bias term and a random error term_

X\_pred **=** [**...**]

_# Normalise its features (except the bias term) with the original means and standard deviations_

X\_pred **=** [**...**]

_# Generate a prediction for this new example_

Y\_pred **=** [**...**]
