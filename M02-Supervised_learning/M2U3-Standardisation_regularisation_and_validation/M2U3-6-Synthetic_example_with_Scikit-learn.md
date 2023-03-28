# **Linear Regression: Synthetic example with Scikit-learn**

M2U3 - Exercise 6

**What are we going to do?**

- We will solve a multivariate linear regression model using Scikit-learn

Remember to follow the instructions for the submission of assignments indicated in [Submission Instructions](https://github.com/Tokio-School/Machine-Learning/blob/main/Instrucciones%20entregas.md).

**Instructions**

Once we developed a hands-on implementation of the multivariate linear regression algorithm with Numpy exclusively, we have been able to see in depth the steps to follow, how the internal mathematical algorithm works, and how all the hyperparameters affect it.

Having a good understanding of how these ML models work, let's see how to use them with the functions of the Scikit-learn ML framework.

In this exercise you will have a blank template with the steps we have followed in previous exercises, which you will have to complete with your code following those steps, but this time using Scikit-learn methods.

In each cell we will suggest a Scikit-learn function that you can use. We won't give you more information about it here, because we want you to look it up for yourself in the documentation: how it works, the algorithms it implements (some of them will be slightly different from the ones we have seen in the course, don't worry as the important thing is the base), arguments, examples, etc.

It sounds like a truism, but I'm sure you will agree with us that the ability to find relevant information in the documentation at all times is extremely important, and it can often cost us a little more than it should :).

Also take the opportunity to dive deeper into the documentation and discover interesting aspects of the framework. We will continue to work with it in subsequent exercises.

In [ ]:

_# TODO: Import all the necessary modules into this cell_

**import** numpy **as** np

**from** matplotlib **import** pyplot **as** plt

**Create a synthetic dataset for linear regression**

- Add a modifiable bias and error term.

In [ ]:

_# TODO: Create a synthetic dataset for linear regression with Scikit-learn_

_# You can use the sklearn.datasets.make\_regression() function_

_# Remember to always use a given random start state to maintain reproducibility_

**Preprocess the data**

- Randomly reorder the data.
- Normalise the data.
- Divide the data into training and test subsets.

_Note_: Why did we use only 2 training and test subsets this time, with no validation subset? Because we will use _k-fold_ for our cross validation.

In [ ]:

_# TODO: Randomly reorder the data_

_# You can use the sklearn.utils.shuffle() function_

In [ ]:

_# TODO: Normalise the examples_

_# You can use the sklearn.preprocessing.StandardScaler() class_

_Note:_ This scaling is equivalent to the basic normalisation we have seen throughout the course. Another more convenient but more complex to comprehend normalisation for more advanced models would be the one implemented in _sklearn.preprocessing.normalize_.

You can find all the available preprocessing classes and functions here: [Sklearn docs: 6.3. Preprocessing data](https://scikit-learn.org/stable/modules/preprocessing.html)

And a graphical comparison: [Compare the effect of different scalers on data with outliers](https://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html)

In [ ]:

_# TODO: Divide the dataset into training and test subsets_

_# You can use the sklearn.model\_selection.train\_test\_split() function_

**Train an initial model**

- Train an initial model on the training subset without regularisation.
- Test the suitability of the model.
- Check if it suffers from deviation or overfitting.

To train a simple multivariate linear regression model, you can use the [sklearn.linear\_model.LinearRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html#sklearn.linear_model.LinearRegression) class

You can consult a complete training example: [Linear Regression Example](https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html#sphx-glr-auto-examples-linear-model-plot-ols-py)

In [ ]:

_# TODO: Train a baseline linear regression model on the training subset without regularisation_

_# Adjust the intercept/bias term and do not normalise the features, as we have already normalised them_

Check the suitability of the model applied to this dataset. To do this you can use:

- The R^2 coefficient of determination method [LinearRegression.score()](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html#sklearn.linear_model.LinearRegression.score)
- The function [sklearn.metrics.mean\_squared\_error](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html#sklearn.metrics.mean_squared_error) (returns the MSE or RMSE)
- Other [regression metrics](https://scikit-learn.org/stable/modules/classes.html#regression-metrics)

Try several of the methods to get to know them better and see their possible differences:

In [ ]:

_# TODO: Test the suitability of the model by evaluating it on the test set_

_# Test 3 of the preceding metrics_

To check whether there might be bias or overfitting, we can calculate e.g., the RMSE on the predictions of the training subset and on those of the test subset:

In [ ]:

_# TODO: Check if the evaluation on both subsets is similar with the RMSE_

**Find the optimal** _ **k-fold** _ **or cross-validation regularisation**

- Train a model for each regularisation value to be considered.
- Train and evaluate them on a K-fold training subset division.
- Choose the optimal model and its regularisation.

We are now going to use a more complex linear regression algorithm, the [sklearn.linear\_model.Ridge](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html#sklearn.linear_model.Ridge) class which allows us to set an L2 regularisation parameter.

In this function, the regularisation argument is called _alpha_, although it should not be confused with the learning rate.

The regularisation we have seen during the course is the one implemented by most Scikit-learn algorithms, its common name being "L2" or "L2-norm".

Consider L2 regularisation parameters in the logarithmic range: 0.1, 0.01, 0.001, 0.0001, etc.

You can follow the link below [K-fold](https://scikit-learn.org/stable/modules/cross_validation.html#k-fold)

In [ ]:

_# TODO: Train a different model for each alpha on a different K-fold_

_# Use a Numpy function to create a logarithmic space of \>5 values between [0, 1e-3]_

alphas **=** [**...**]

_# Create k K-fold splits_

kfolds **=** [**...**]

_# Iterate over the splits for your models and evaluate them on the generated CV subset_

linear\_models **=** []

best\_model **=**** None**

**for** train, cv **in** kfolds **.** split(X):

_#Train a model on the training subset_

_# Remember to set the corresponding alpha/regularisation parameter, adjust the bias, and do not normalise_

_# Evaluate it on the CV subset using its model.score()_

_# Save the model with the best score for the best\_model variable and display the alpha of the best model_

alpha **=** [**...**]

print('L2 regularization:', alpha)

model **=** [**...**]

linear\_models **.** append(model)

_# If the model is better than the best model so far..._

best\_model **=** [**...**]

print('L2 regularisation of the best model so far:', alpha)

**Finally, evaluate the model on the test subset**

- Display the coefficients and intercept of the best model.
- Evaluate the best model on the initial test subset.
- Calculate the residuals on the test subset and plot them.

In [ ]:

_# TODO: Evaluate the model on the initial test subset._

_# Display the coefficients and intercept of the best trained model._

print('Intercept coefficients of the trained model:')

print([...]) _# Display the intercept as the first coefficient_

_# Make predictions about the test subset_

y\_test\_pred **=** [**...**]

_# Calculate the model evaluation metrics: RMSE and coefficient of determination R^2_

rmse **=** [**...**]

r2\_score **=** [**...**]

print('Root mean square error(RMSE): %.2f' **%** rmse)

print('Coefficient of determination: %.2f' **%** r2\_score)

_# Calculate the residuals on the test subset_

residuals **=** [**...**]

_# Plot them graphically_

plt **.** figure(1)

_# Fill in your code_

plt **.** show()

**Make predictions about new examples**

- Generate a new example, following the same pattern as the original dataset.
- Normalise its features.
- Generate a prediction for this new example.

In [ ]:

_# TODO: Make predictions about a new manually created example_

_# Create the new example_

X\_pred **=** [**...**]

_# Normalise its features_

X\_pred **=** [**...**]

_# Generate a prediction for this new example_

y\_pred **=** [**...**]
