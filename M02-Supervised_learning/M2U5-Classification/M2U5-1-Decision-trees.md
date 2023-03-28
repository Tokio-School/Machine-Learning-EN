# **Decision Trees: Scikit-Learn**

M2U5 - Exercise 1

**What are we going to do?**

- We will train a linear regression model using decision trees
- We will check to see if there is any deviation or overfitting in the model
- We will optimise the hyperparameters with validation
- We will evaluate the model on the test subset

Remember to follow the instructions for the submission of assignments indicated in [Submission Instructions](https://github.com/Tokio-School/Machine-Learning/blob/main/Instrucciones%20entregas.md).

**Instructions**

We are going to solve a multivariate linear regression problem similar to the previous exercises, but this time using a decision tree for linear regression.

An example that you can use as a reference for this exercise: [Decision Tree Regression](https://scikit-learn.org/stable/auto_examples/tree/plot_tree_regression.html)

In [ ]:

_# TODO: Import all the necessary modules into this cell_

**Generate a synthetic dataset**

Generate a synthetic dataset with a fairly large error term and few features, manually or with Scikit-learn:

In [ ]:

_# TODO: Generate a synthetic dataset, with few features and a significant error term_

_# Do not add a bias term to X_

m **=** 1000

n **=** 2

X **=** [**...**]

Theta\_true **=** [**...**]

error **=** 0.3

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

In [ ]:

_# TODO: Graphically represent the dataset in 3D to ensure that the error term is sufficiently high_

plt **.** figure(1)

plt **.** title()

plt **.** xlabel()

plt **.** ylabel()

[**...**]

plt **.** show()

**Preprocess the data**

- Randomly reorder the data.
- Normalise the data.
- Divide the dataset into training and test subsets.

_Note:_ We will use K-fold again for the cross-validation.

In [ ]:

_# TODO: Randomly reorder the data, normalise the examples, and divide them into training and test subsets_

**Train an initial model**

We will begin exploring decision tree models for regression with an initial model.

To do this, train a [sklearn.tree.DecisionTreeRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html) model on the training subset:

In [ ]:

_# TODO: Train a regression tree on the training subset with a max. depth of 2_

Now check the suitability of the model by evaluating it on the test subset:

In [ ]:

_# TODO: Evaluate the model with MSE, RMSE and R^2 on the test subset_

y\_test\_pred **=** [**...**]

mse **=** [**...**]

rmse **=** [**...**]

r2\_score **=** [**...**]

print('mean square error: {%.2f}' **.** format(mse))

print('Root mean square error: {%.2f}' **.** format(rmse))

print('Coefficient of determination: {%.2f}' **.** format(r2\_score))

_QUESTION: Do you think there is deviation or overfitting in this model?_

To find out, compare its accuracy with that calculated on the training subset and answer in this cell:

In [ ]:

_# TODO: Now evaluate the model with MSE, RMSE and R^2 on the training subset_

y\_train\_pred **=** [**...**]

mse **=** [**...**]

rmse **=** [**...**]

r2\_score **=** [**...**]

print('mean square error: {%.2f}' **.** format(mse))

print('Root mean square error: {%.2f}' **.** format(rmse))

print('Coefficient of determination: {%.2f}' **.** format(r2\_score))

As mentioned above, decision trees tend to overfit, to over-adjust to the data used to train them, and sometimes fail to predict well on new examples.

We are going to check this graphically by training another model with a much larger maximum depth of 6:

In [ ]:

_# TODO: Train another regression tree on the training subset with max. depth of 6_

In [ ]:

_# TODO: Now evaluate the model with MSE, RMSE, and R^2 on the training subset_

y\_train\_pred **=** [**...**]

mse **=** [**...**]

rmse **=** [**...**]

r2\_score **=** [**...**]

print('mean square error: {%.2f}' **.** format(mse))

print('Root mean square error: {%.2f}' **.** format(rmse))

print('Coefficient of determination: {%.2f}' **.** format(r2\_score))

Compare the training accuracy of this model with the previous one (on the training subset).

_QUESTION_: Is the accuracy greater or lesser as the maximum depth of the tree increases?

We will now plot both models, to check whether they suffer from deviation or overfitting.

To do so, you can be guided by the preceding example: [Decision Tree Regression](https://scikit-learn.org/stable/auto_examples/tree/plot_tree_regression.html)

In [ ]:

_# TODO: Graphically represent the predictions of both models_

plt **.** figure(2)

plt **.** title([**...**])

plt **.** xlabel([**...**])

plt **.** ylabel([**...**])

_# Plot the training subset for features 1 and 2 (with different shapes) in a dot plot_

plt **.** scatter([**...**])

plt **.** scatter([**...**])

_# Plot the test subset for features 1 and 2 (with different shapes) in a dot plot, with a different colour from the training subset_

plt **.** scatter([**...**])

plt **.** scatter([**...**])

_# Plot the predictions of the two models on a line graph, with different colours, and a legend to distinguish them_

_# Use a linear space with a large number of elements between the max. and min. value of both X features as the horizontal axis._

x\_axis **=** [**...**]

plt **.** plot([**...**])

plt **.** plot([**...**])

plt **.** show()

As we have seen, too small a max. depth generally leads to a model with deviation, a model that is not able to fit the curve well enough, while too high a max. depth leads to a model with overfitting, a model that fits the curve very well, but does not have good accuracy on future examples.

Therefore, among all regression tree hyperparameters, we have the maximum depth, which we need to optimise using validation. There are also other hyperparameters, such as the criteria for measuring the quality of a split, the strategy for creating that split, the min. number of examples needed to split a node, etc.

For the sake of simplicity, let's start by performing a cross-validation just to find the optimal value for the maximum depth:

In [ ]:

_# TODO: Train a different model for each max\_depth value considered on a different fold_

# _Values of max\_depth to be considered in an integer space [1, 8]_

max\_depths **=** [**...**]

print('Max. depths to be considered:')

print(max\_depths)

_# Create x K-fold splits, one for each value of max\_depth to be considered_

kf **=** [**...**]

_# Iterate on the splits, train your models and evaluate them on the generated CV subset_

linear\_models **=** []

best\_model **=**** None**

**for** train, cv **in** kf **.** split(X):

_#Train a model on the training subset_

_# Evaluate it on the cv subset using its method score()_

_# Save the model with the best score on the best\_model variable and display the alpha of the best model_

alpha **=** [**...**]

print('Max. depth used:', max\_depth)

linear\_models **.** append([**...**])

_# If the model is better than the best model so far, update the best model found._

best\_model **=** [**...**]

print('Max. depth and R^2 of the best tree so far:', max\_depth, best\_model.score([...]))

**Evaluate the model on the test subset**

Finally, we are going to evaluate the model on the test subset.

To do this, calculate its MSE, RMSE, and R^2 metrics and plot the model predictions and residuals vs. the test subset:

In [ ]:

_# TODO: Evaluate the model with MSE, RMSE and R^2 on the test subset_

y\_train\_test **=** [**...**]

mse **=** [**...**]

rmse **=** [**...**]

r2\_score **=** [**...**]

print('mean square error: {%.2f}' **.** format(mse))

print('Root mean square error: {%.2f}' **.** format(rmse))

print('Coefficient of determination: {%.2f}' **.** format(r2\_score))

In [ ]:

_# TODO: Plot the predictions of the best tree on the test subset and its residuals_

plt **.** figure(3)

plt **.** title([**...**])

plt **.** xlabel([**...**])

plt **.** ylabel([**...**])

_# Plot the test subset on a dot plot, representing both features with different shapes_

plt **.** scatter([**...**])

_# Plot the model's predictions on a line graph_

_# Use a linear space with a large number of elements between the max. and min. value of the X\_test features as the horizontal axis._

x\_axis **=** [**...**]

plt **.** plot([**...**])

_# Calculate the residuals and plot them as a bar chart on the horizontal axis._

residuals **=** [**...**]

plt **.** bar([**...**]

plt **.** show()
