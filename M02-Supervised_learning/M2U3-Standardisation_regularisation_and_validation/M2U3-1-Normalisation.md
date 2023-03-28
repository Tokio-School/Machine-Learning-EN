# **Linear Regression: Normalisation**

M2U3 - Exercise 1

**What are we going to do?**

- We will create a synthetic dataset with features in different value ranges
- We will train a linear regression model on the original dataset
- We will normalise the original dataset
- We will train another linear regression model on the normalised dataset
- We will make a comparison between the training of both models, normalised and non-normalised

Remember to follow the instructions for the submission of assignments indicated in [Submission Instructions](https://github.com/Tokio-School/Machine-Learning/blob/main/Instrucciones%20entregas.md).

In [ ]:

**import** time

**import** numpy **as** np

**from** matplotlib **import** pyplot **as** plt

**Creation of a synthetic dataset**

We are going to manually create a synthetic dataset for linear regression.

Create a synthetic dataset with an error term of 10% of the value over_Y_ and an _X_ approx. in the range (-1, 1), this time manually, not with the specific Scikit-learn methods, with the code used in previous exercises:

In [ ]:

_# TODO: Copy code from previous exercises to generate a dataset with a bias term and an error term_

m = 1000

n = 4

X **=** [**...**]

Theta\_true **=** [**...**]

error **=** 0.1

Y **=** [**...**]

In [ ]:

_# Check the values and dimensions of the vectors_

print('Theta and its dimensions to be estimated:')

print()

print()

print('First 10 rows and 5 columns of X and Y:')

print()

print()

print('Dimensions of X and Y:')

print()

We will now modify the dataset to ensure that each feature, each column of _X_, has a different order of magnitude and mean.

To do this, multiply each column of _X_ (except the first one, the bias, which must be all 1's) by a different range and add a different bias value to it.

The value we then add up will be the mean of that feaure or column, and the value by which we multiply its range or scale.

E.g., X1=X1∗10^3+3.1415926, where 10^3 would be the mean and 3.1415926 the scale of the feature.

In [ ]:

_# TODO: For each column of X, multiply it by a range of values and add a different mean to it_

_# The arrays of ranges and averages must be of length n_

_# Create an array with the ranges of values, e.g.: 1e0, 1e3, 1e-2, 1e5_

ranges **=** [**...**]

averages **=** [**...**]

X **=** [**...**]

print('X with different averages and scales')

print(X)

print(X **.** shape)

Remember that you can run Jupyter cells in a different order from their position in the document. The brackets to the left of the cells will mark the order of execution, and the variables will always keep their values after the last executed cell, so **be careful!**

**Training and evaluation of the model**

Once again, we will train a multivariate linear regression model. This time, we are going to train it first on the original, non-normalised dataset, and then retrain it on the normalised dataset, in order to compare both models and training processes and see the effects of normalisation.

To do this you must copy the cells or code from previous exercises and train a multivariate linear regression model, optimized by gradient descent, on the original dataset.

You must also copy the cells that test the training of the model, representing the cost function vs. the number of iterations.

You do not need to make predictions about this data or evaluate the model's residuals. In order to compare them, we will do so only on the basis of the final cost.

In [ ]:

_# TODO: Train a linear regression model and plot the evolution of its cost function_

_# Use the non-normalised X_

_# Add the suffix "\_no\_norm" to the Theta and j\_hist variables returned by your model_

**Data normalisation**

We are going to normalise the data from the original dataset.

To do this, we are going to create a normalisation function that applies the necessary transformation, according to the formula:

xj=xj−μjσj

In [ ]:

_# TODO: Implement a normalisation function to a common range and with a mean of 0_

**def** normalize(x, mu, std):

""" Normalise a dataset with X examples

Positional arguments:

x -- Numpy 2D array with the examples, no bias term

mu -- Numpy 1D vector with the mean of each feature/column

std -- Numpy 1D vector with the standard deviation of each feature/column

Return:

x\_norm -- Numpy 2D array with the examples, and their normalised features

"""

**return** [**...**]

In [ ]:

_# TODO: Normalise the original dataset using your normalisation function_

_# Find the mean and standard deviation of the features of X (columns), except the first column (bias)._

mu **=** [**...**]

std **=** [**...**]

print('original X:')

print(X)

print(X **.** shape)

print('Mean and standard deviation of the features')

print(mu)

print(mu **.** shape)

print(std)

print(std **.** shape)

print('normalised X:')

X\_norm **=** np **.** copy(X)

X\_norm[**...**] **=** normalise(X[**...**], mu, std) _# Normalise only column 1 and the subsequent columns, not column 0_

print(X\_norm)

print(X\_norm **.** shape)

_BONUS:_

1. Calculate the means and standard deviations of _X\_norm_ according to its features/columns.
2. Compare them with those of _X, mu,_ and _std_
3. Plot the distributions of _X_ and _X\_norm_ in a bar graph or box plot (you can use multiple Matplotlib subplots).

**Retraining the model and comparison of results**

Now retrain the model on the normalised dataset. Check the final cost and the iteration at which it converged.

To do this, you can go back to the training cells of the model and check the evolution of the cost function and modify the _X_ used for _X\_norm_.

In many cases, because it is such a simple model, there may be no noticeable improvement. Depending on the capacity of your working environment, try using a higher number of features and slightly increasing the error term of the dataset.

In [ ]:

_# TODO: Train a linear regression model and plot the evolution of its cost function_

_# Use the normalised X_

_# Add the suffix "\_norm" to the Theta and j\_hist variables returned by your model_

_QUESTION: Is there any difference in the accuracy and training time of the model on non-normalised data and the model on normalised data? If you increase the error term and the difference in means and ranges between the features, does it make more of a difference?_

**Beware of the original Theta**

For the original dataset, before normalisation, the relationship Y=X×Θ was fulfilled.

However, we have now modified the _X_ term of this function.

Therefore, check what happens if you want to recompute _Y_ using the normalized _X_:

In [ ]:

_# TODO: Check for differences between the original Y and the Y computed using the normalized X_

_# Check the value of Y by multiplying X\_norm and Theta\_true_

Y\_norm **=** [**...**]

_# Check for differences between Y\_norm and Y_

diff **=** Y\_norm **-** Y

print('Difference between Y\_norm and Y (first 10 rows):')

print(diff[:10])

# Plot the difference between the_Y's_ vs _X_ on a dot plot

[**...**]

**Make predictions**

Similarly, what happens when we are going to use the model to make predictions?

Generate a new dataset _X\_pred_ following the same method you used for the original _X_ dataset, incorporating the bias term, multiplying its features by a range and adding different values to them, without finally normalising the dataset.

Also calculate its _Y\_pred\_true_ (without error term), as the true value of _Y_ to try to predict:

In [ ]:

_# TODO: Generate a new dataset with fewer examples and the same number of features as the original dataset_

_# Make sure it has a normalized mean or range across features/columns_

X\_pred **=** [**...**]

Y\_pred\_true **=** np **.** matmul(X\_pred, Theta\_true)

Now check if there is any difference between the _Y\_pred\_true_ and the _Y\_pred_ that your model predicts:

In [ ]:

_# TODO: Check the differences between the actual Y and the predicted Y_

Y\_pred **=** np **.** matmul(X\_pred, theta)

diff **=** Y\_pred\_true **-** Y\_pred

print('Differences between actual Y and predicted Y:')

print(diff[:10])

Since the predictions are not correct otherwise, we should first normalise the new _X\_pred_ before generating the predictions:

In [ ]:

_# TODO: Normalise the X\_pred_

X\_pred[**...**] **=** normalize(X\_pred[**...**], mu, std)

print(X\_pred[:10,:])

print(X\_pred **.** shape)

This time we have not generated a new, different variable by normalisation, but it remains the variable _X\_pred_.

You can then rerun the previous cell to, now that _X\_pred_ is normalised, check if there is any difference between the actual_Y_ and the predicted_Y_.

So always remember:

- The _theta_ calculated when training the model will always be relative to the normalised dataset, and cannot be used for the original dataset, since with the same_Y_ and a different _X_, _Theta_ must change.
- To make predictions on new examples, we first have to normalise them as well, using the same values for the means and standard deviations that we originally used to train the model.
