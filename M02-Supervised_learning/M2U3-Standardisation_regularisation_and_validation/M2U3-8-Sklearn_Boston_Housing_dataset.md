# **Linear Regression with Scikit-learn: Boston Housing Dataset**

M2U3 - Exercise 8

**What are we going to do?**

- We will analyse the Scikit-learn Boston Housing real estate sample dataset.
- We will train a multivariate linear regression model on the dataset

Remember to follow the instructions for the submission of assignments indicated in [Submission Instructions](https://github.com/Tokio-School/Machine-Learning/blob/main/Instrucciones%20entregas.md).

# **Linear Regression: Scikit-learn on the Boston Housing dataset**

**What are we going to do?**

- We will analyse the Scikit-learn Boston Housing real estate sample dataset.
- We will train a multivariate linear regression model on the dataset

Once again, we are going to fully train another linear regression model using Scikit-learn, and since we are generally going to follow the same steps as in the previous exercise, we are going to minimise the prompts so as not to distract you.

For this exercise you can use the following references, among others:

- [Boston Housing dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_boston.html#sklearn.datasets.load_boston)

_NOTE:_ This function and dataset will be deprecated in Scikit-learn v1.2.

In [ ]:

_# TODO: Import all the necessary modules into this cell_

**Load the Boston Housing dataset**

Before starting to work with the dataset, analyse its features and some of the examples included.

In [ ]:

_# TODO: Load the Boston Housing dataset, analyse its features and examples and finally load it as_

_# an (X, Y) tuple_

**Preprocess the data**

- Randomly reorder the data
- Normalise the data
- Divide the dataset into training and test subsets

_Note:_ Before normalising the data from a new dataset, check whether it is necessary and ensure it has not already been normalised.

In [ ]:

_# TODO: Randomly reorder and normalise the data and split the dataset into 2 subsets, as needed_

**Train an initial model**

- Train an initial model on the training subset without regularisation.
- Test the suitability of the model.
- Check if there is any deviation or overfitting.

If so, revert to using a linear regression model, such as the [Lasso](https://scikit-learn.org/stable/modules/linear_model.html#lasso) (without regularisation):

In [ ]:

_# TODO: Train a simpler linear regression model on the training subset without regularisation._

In [ ]:

_# TODO: Test the suitability of the model by evaluating it on the test set with several metrics_

In [ ]:

_# TODO: Check if the evaluation on both subsets is similar with the RMSE_

**Train the model with CV**

- Train a model for each regularisation value to be considered.
- Train and evaluate them on the training subset using K-fold.
- Choose the optimal model and its regularisation.

Train the model using the Lasso [algorithm](https://scikit-learn.org/stable/modules/linear_model.html#lasso) and optimise the regularisation using GridSearchCV:

In [ ]:

_# TODO: Train a different model for each alpha on a different K-fold, evaluate them and select_

_# the most accurate model using GridSearchCV_

**Finally, evaluate the model on the test subset**

- Display the coefficients and RMSE of the best model.
- Evaluate the best model on the initial test subset.
- Calculate the residuals for the test subset and plot them.

In [ ]:

_# TODO: Evaluate the best model on the initial test subset and calculate its residuals_
