# **Linear Regression: Validation, final evaluation, and metrics**

M2U3 - Exercise 4

**What are we going to do?**

- Create a synthetic dataset for multivariate linear regression
- Preprocess the data
- We will train the model on the training subset and check its suitability
- We will find the optimal _lambda_ hyperparameter for the validation subset
- We will evaluate the model on the test subset
- We will make predictions about new future examples

Remember to follow the instructions for the submission of assignments indicated in [Submission Instructions](https://github.com/Tokio-School/Machine-Learning/blob/main/Instrucciones%20entregas.md).

In [ ]:

**import** time

**import** numpy **as** np

**from** matplotlib **import** pyplot **as** plt

**Create a synthetic dataset for linear regression**

We will start, as usual, by creating a synthetic dataset for this exercise.

This time, for the error term, use a non-symmetric range, different from [-1, 1], such as [-a, b], with parameters _a_ and _b_ that you can control. In this way we can modify this distribution at later points to force a greater difference between the training and validation subsets.

In [ ]:

_# TODO: Generate a synthetic dataset manually, with a bias term and an error term_

m = 1000

n = 3

X **=** [**...**]

Theta\_true **=** [**...**]

error **=** 0.2

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

**Preprocess the data**

We will preprocess the data completely, to leave it ready to train the model.

To preprocess the data, we will follow the steps below:

- Randomly rearrange the data.
- Normalise the data.
- Divide the dataset into training, validation, and test subsets.

**Randomly rearrange the dataset**

This time we are going to use a synthetic dataset created using random data. Therefore, it will not be necessary to rearrange the data, as it is already randomized and disorganized by default.

However, we may often encounter real datasets whose data has a certain order or pattern, which can confound our training.

Therefore, before starting to process the data, the first thing we need to do is to randomly reorder it, especially before splitting it into training, validation, and test subsets.

_Note:_ **Very important!** Remember to always reorder the _X_ and _Y_ examples and results in the same order, so that each example is assigned the same result before and after reordering.

In [ ]:

_# TODO: Randomly reorder the dataset_

print('First 10 rows and 5 columns of X and Y:')

print()

print()

_# Use an initial random state of 42, in order to maintain reproducibility_

print('Reorder X and Y:')

X, Y **=** [**...**]

print('First 10 rows and 5 columns of X and Y:')

print()

print()

print('Dimensions of X and Y:')

print()

Check that X and Y have the correct dimensions and a different order than before.

**Normalise the dataset**

Once the data has been randomly reordered, we will proceed with the normalisation of the _X_ examples dataset.

To do this, copy the code cells from the previous exercises to normalise it.

_Note:_ In previous exercises we used 2 different code cells, one to define the normalisation function and one to normalise the dataset. You can combine both cells into one cell to save this preprocessing in a reusable cell for future use.

In [ ]:

_# TODO: Normalise the dataset with a normalisation function_

**def** normalize(x, mu **=None** , std **=None** ):

""" Normalises a dataset with X examples

Positional arguments:

x -- Numpy 2D array with the examples, without bias term

mu -- Numpy 1D vector with the mean of each feature/column

std -- Numpy 1D vector with the standard deviation of each feature/column

Returns:

x\_norm -- 2D ndarray with the examples, and their normalised features

mu, std -- if mu and std are None, compute and return those parameters. If not, use these parameters to normalise x without returning them

"""

**return** [**...**]

_# Find the mean and standard deviation of the X features (columns), except for the first one (bias)_

mu **=** [**...**]

std **=** [**...**]

print('Original X (first 10 rows and columns):')

print(X[:10, :10])

print(X **.** shape)

print('Mean and standard deviation of the features:')

print()

print(mu **.** shape)

print(std)

print(std **.** shape)

print('Normalised X:')

X\_norm **=** np **.** copy(X)

X\_norm[**...**] **=** normalize(X[**...**], mu, std) _# Normalizstion only for column 1 and the subsequent columns, not for column 0_

print(X\_norm)

print(X\_norm **.** shape)

**Divide the dataset into training, validation, and test subsets**

Finally, we will divide the dataset into the 3 subsets to be used.

For this purpose, we will use a ratio of 60%/20%/20%, as we start with 1000 examples. As we said, for a different number of examples, we can modify the ratio:

In [ ]:

_# TODO: Divide the X and Y dataset into the 3 subsets according to the indicated ratios_

ratio **=** [60,20,20]

print('Ratio:\n', ratio, ratio[0] **+** ratio[1] **+** ratio[2])

_# Calculate the cutoff indices for X and Y_

_# Tip: the round() function and the x.shape attribute may be useful to you_

r **=** [0, 0]

r[0] **=** [**...**]

r[1] **=** [**...**]

print('cutoff indices:\n', r)

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

**Train an initial model on the training subset**

Before we begin optimizing the _lambda_ hyperparameter, we will train an initial unregularized model on the training subset to check its performance and suitability, and to be sure that it makes sense to train a multivariate linear regression model on this dataset, as its features might not be suitable; there might be low correlation between them, they might not follow a linear relationship, etc.

To do this, follow these steps:

- Train an initial model, without regularization, with _lambda_ at 0.
- Plot the history of the cost function to check its evolution.
- Retrain the model if necessary, e.g., by varying the learning rate _alpha_.

Copy the cells from previous exercises in which you implemented the regularized cost and gradient descent functions, and copy the cell where you trained the model:

In [ ]:

_# TODO: Copy the cells with the regularised cost and gradient descent functions_

In [ ]:

_# TODO: Copy the cell where we trained the previous model_

_# Train your model on the unregularised training subset and get the final cost and the history of its evolution_

Check the training of the model as in previous exercises, plotting the evolution of the cost function versus the number of iterations, and copying the corresponding code cell:

In [ ]:

_# TODO: Plot the evolution of the cost function vs. the number of iterations_

plt **.** figure(1)

As we said before, review the training of your model and modify some parameters to retrain it to improve its performance, if necessary: the learning rate, the convergence point, the maximum number of iterations, etc., except for the _lambda_ regularisation parameter, which must be set to 0.

_Note:_ This point is important, as these hyperparameters will generally be the same ones that we will use for the remainder of the optimisation of the model, so now is the time to find the right values.

**Check for deviation or overfitting,** _ **bias** _ **or** _ **variance** _

There is a test we can quickly do to check whether our initial model clearly suffers from deviation, variance, or has a more or less acceptable performance.

We will plot the evolution of the cost function of 2 models, one trained on the first _n_ examples of the training subset and the other trained on the first _n_ examples of the validation subset.

Since the training subset and the validation subset are not the same size, use only the same number of examples from the training subset as the total number of examples in the validation subset.

To do this, train 2 models under equal conditions by copying the corresponding code cells again:

In [ ]:

_# TODO: Establish a common theta\_ini and hyperparameters for both models, to train them both on equal terms_

theta\_ini **=** [**...**]

print('Theta initial:')

print(theta\_ini)

alpha **=** 1e-1

lambda\_ **=** 0.

e **=** 1e-3

iter\_ **=** 1e3

print('Hyperparameters used:')

print('Alpha:', alpha, 'Error máx.:', e, 'Nº iter', iter\_)

In [ ]:

_# TODO: Train a model without regularisation on the first n values of X\_train, where n is the no. of_

_# examples available in X\_val_

_# Use j\_hist\_train and theta\_train as variable names to distinguish them from the other model_

_Note:_ Check that _theta\_ini_ has not been modified, or modify your code so that both models use the same _theta\_ini_.

In [ ]:

_# TODO: In the same way, train a model without regularisation on X\_val with the same parameters_

_# Remember to use j\_hist\_val and theta\_val as variable names_

Now plot both evolutions on the same graph, with different colours, so that they can be compared:

In [ ]:

_# TODO: Plot the cost evolution in both datasets on a line graph for comparison_

plt **.** figure(2)

plt **.** title()

plt **.** xlabel()

plt **.** ylabel()

_# Use different colours for both series, and provide a legend to distinguish them_

plt **.** plot()

plt **.** plot()

plt **.** show()

With a random synthetic dataset it is difficult to overfit, as the original data will follow the same pattern, but by proceeding in this way we will be able to identify the following problems:

- If the final cost in both subsets is high, there may be a problem with deviation or _bias_.
- If the final cost for both subsets is very different from each other, there may be a problem with overfitting or _variance_, especially when the cost for the training subset is much lower than the cost for the validation subset.

Recall the significance of deviation and overfitting:

- Deviation occurs when the model cannot fit the curve of the dataset well enough, either because the features are not correct (or others are missing), or because the data has too much error, or because the model follows a different relationship, or is too simple.
- Overfitting occurs when the model fits the dataset curve very well, too well, too closely to the examples on which it has been trained, and when it has to predict on new outcomes it does not do so correctly.

**Test the suitability of the model**

As mentioned above, another reason to train an initial model is to check whether it makes sense to train a multivariate linear regression model on such a dataset.

If we see that the model suffers from overfitting, we can always correct it with regularisation. However, if we see that it suffers from high deviation, i.e., that the final cost is very high, it may be that our type of model or the features chosen are not suitable for this problem.

In this case, we found that the error is low enough to make further training of this multivariate linear regression model promising.

**Find the optimal** _ **lambda** _ **hyperparameter on the validation subset**

Now, in order to find the optimal _lambda_, we will train a different model for each _lambda_ value to be considered on the training subset, and check its accuracy on the validation subset.

We will plot the final error or cost of each model vs. the _lambda_ value used, to see which model has a lower error or cost on the validation subset.

In this way, we train all models on the same subset and under equal conditions (except _lambda_), and we evaluate them on a subset of data they have not seen previously, which has not been used to train them.

The validation subset is therefore not used to train the model, but only to evaluate the optimal _lambda_ value. Except for the previous point, where we made a quick initial assessment of the possible occurrence of overfitting.

In [ ]:

_# TODO: Train a model for each different lambda value on X\_train and evaluate it on X\_val._

lambdas **=** [0., 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1e0, 3e0, 1e1]

_# BONUS: Generate an array of lambdas with 10 values on a logarithmic scale between 10^-3 and 10, alternating between values whose first non-zero decimal is a 1 or a 3, like this list_

_# Complete the code to train a different model for each value of lambda on X\_train_

_# Store your theta and final error/cost_

_# Afterwards, evaluate its total cost on the validation subset_

_# Store this information in the following arrays, of the same size as the lambda arrays_

j\_train **=** [**...**]

j\_val **=** [**...**]

theta\_val **=** [**...**]

Once all models have been trained, on a line graph plot their final cost on the training subset and the final cost on the validation subset vs. the _lambda_ value used:

In [ ]:

_# TODO: Plot the final error for each value of lambda_

plt **.** figure(3)

_# Fill in your code_

Once these final errors are plotted, we can automatically choose the model with the optimal _lambda_ value:

In [ ]:

_# TODO: Choose the optimal model and lambda value, with the lowest error on the validation subset_

_# Iterate over the theta and lambda of all the models and choose the one with the lowest cost on the validation subset_

j\_final **=** [**...**]

theta\_final **=** [**...**]

lambda\_final **=** [**...**]

Once all the above steps have been implemented, we have our trained model and its hyperparameters optimised.

**Finally, evaluate the model on the test subset**

Finally, we have found our optimal _theta_ and _lambda_ hyperparameter coefficients, so we now have a trained, ready-to-be-used model.

However, although we have calculated its error or final cost on the validation subset, we have used this subset to select the model or to "finish training" it. Therefore, we have not yet tested how this model will work on data it has never seen before.

To do this, we will finally evaluate it on the test subset, on a subset that we have not yet used to either train the model or to select its hyperparameters. A separate subset that the model training has not yet seen.

Therefore, we will calculate the total error or cost on the test subset and graphically check the residuals of the model on it:

In [ ]:

_# TODO: Calculate the error of the model on the test subset using the cost function with the corresponding_

_# theta and lambda_

j\_test **=** [**...**]

In [ ]:

_# TODO: Calculate the predictions of the model on the test subset, its residuals, and plot them_

Y\_test\_pred **=** [**...**]

residuals **=** [**...**]

plt **.** figure(4)

_# Fill in your code_

plt **.** show()

In this way we can get a more realistic idea of how accurate our model is and how it will behave with new examples in the future.

**Make predictions about new future examples**

With our model trained, optimised, and evaluated, all that remains is to put it to work by making predictions with new examples.

To do this, we will:

- Generate a new example, following the same pattern as the original dataset.
- Normalise its features before predictions can be made about them.
- Generate a prediction for this new example.

In [ ]:

_# TODO: Generate a new example following the original pattern, with a bias term and a random error term_

X\_pred **=** [**...**]

_# Normalise its features (except the bias term) to the original means and standard deviations_

X\_pred **=** [**...**]

_Generate a prediction for this new example._

Y\_pred **=** [**...**]

**Data Preprocessing with Scikit-learn**

Finally, find and use the functions available in Scikit-learn to preprocess data:

1. [Randomly reordering](https://scikit-learn.org/stable/modules/generated/sklearn.utils.shuffle.html?highlight=shuffle#sklearn.utils.shuffle)
2. [Normalising/scaling](https://scikit-learn.org/stable/modules/preprocessing.html#standardization-or-mean-removal-and-variance-scaling)
3. [Dividing the data into the 3 corresponding subsets](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html?highlight=split#sklearn.model_selection.train_test_split)

In [ ]:

_# TODO: Use Scikit-learn functions to randomly reorder, normalise, and split the data into training, validation, and test subsets_

_# Use the original X instead of X\_norm_

X\_reord **=** [**...**]

X\_scaled **=** [**...**]

X\_train, X\_val, X\_test, Y\_train, Y\_val, Y\_test **=** [**...**]

BONUS: Can you correct your code to apply these standard operations in as few lines as possible and leave it ready for reuse every time?
