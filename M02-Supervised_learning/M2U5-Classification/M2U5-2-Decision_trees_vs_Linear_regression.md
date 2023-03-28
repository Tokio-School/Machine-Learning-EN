# **Decision trees vs. Linear regression**

M2U5 - Exercise 2

**What are we going to do?**

- We will compare the accuracy and behaviour of decision trees versus traditional linear regression

Remember to follow the instructions for the submission of assignments indicated in [Submission Instructions](https://github.com/Tokio-School/Machine-Learning/blob/main/Instrucciones%20entregas.md).

**Instructions**

It is sometimes felt that regression trees may not be as accurate and fall into more overfitting when compared to traditional linear regression, especially when there are a high number of features.

In this exercise, we will follow the usual steps to train 2 linear regression models: a decision tree, and a Lasso.

In [1]:

_# TODO: Import all the necessary modules into this cell_

**Generate a synthetic dataset**

Generate a synthetic dataset with a fairly large error term and few features, manually or with Scikit-learn:

In [ ]:

_# TODO: Create a synthetic dataset with few features and a significant error term_

_# Do not add a bias term to X_

m **=** 1000

n **=** 9

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

**Preprocess the data**

- Randomly reorder the data.
- Normalise the data.
- Divide the dataset into training and test subsets.

_Note:_ We will use K-fold again for the cross-validation.

In [ ]:

_# TODO: Randomly reorder the data, normalise the examples, and divide them into training and test subsets_

**Optimise the models using cross-validation**

- Train a model for each regularisation value or max. depth to be considered.
- Train and evaluate them on a K-fold training subset division.
- Choose the optimal model and its regularisation.

Consider similar parameters to those from previous exercises:

- Maximum depth in the range [1, 8]
- L2 _alpha_ regularisation parametersin the logarithmic range [0, 0.1]: 0.1, 0.01, 0.001, 0.0001, etc.

You can copy the cells from previous exercises and modify them

In [ ]:

_# TODO: Train a different model on a different K-fold for the regression tree and the Lasso_

_# Iterate over the necessary K-fold splits, train your models and evaluate them on the CV subset_

best\_tree **=** [**...**]

best\_lasso **=** [**...**]

**Evaluate the model on the test subset**

Finally, we are going to evaluate the best decision tree and Lasso model on the test subset.

To do this, calculate their MSE, RMSE, and R^2 score metrics and plot the model predictions and residuals vs. the test subset:

In [ ]:

_# TODO: Evaluate the model with MSE, RMSE, and R^2 on the test subset for the best tree and Lasso_

y\_train\_test **=** [**...**]

mse **=** [**...**]

rmse **=** [**...**]

r2\_score **=** [**...**]

print('mean square error: {%.2f}' **.** format(mse))

print('Root mean square error: {%.2f}' **.** format(rmse))

print('Coefficient of determination: {%.2f}' **.** format(r2\_score))

Finally, check their possible deviation or overfitting and final accuracy by plotting the residuals of both models:

In [ ]:

_# TODO: Plot the residuals of both models as line graphs with different colours, vs. the index of the examples (m)_

tree\_residuals **=** [**...**]

lasso\_residuals **=** [**...**]

plt **.** figure(3)

plt **.** title([**...**])

plt **.** xlabel([**...**])

plt **.** ylabel([**...**])

plt **.** plot([**...**])

plt **.** show()

_Are there significant differences between the two models? What happens if we vary the error or the number of features in the original dataset, how do both types of models respond?_

In the case of the regression tree, we may not have made the fairest comparison, since there are still other hyperparameters that we can modify: [sklearn.tree.DecisionTreeRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html)

_ **Bonus:** _ **Optimisation of all the decision tree hyperparameters**

_Do you have the courage to use _[_sklearn.model\_selection.GridSearchCV_](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)_ not only to optimise max\_depth, but for all the hyperparameters of the regression tree?_

There is an example on the GridSearchCV documentation page you can refer to.
