# **Logistic Regression: Scikit-learn on the Iris dataset**

M2U5 - Exercise 8

**What are we going to do?**

- We will download the Iris dataset
- We will preprocess the dataset using Scikit-learn methods
- We will train a multiclass classification model using Scikit-learn

Remember to follow the instructions for the submission of assignments indicated in [Submission Instructions](https://github.com/Tokio-School/Machine-Learning/blob/main/Instrucciones%20entregas.md).

**Instructions**

We will now solve the same model using Scikit-learn methods.

You can use the following example as a reference for this exercise: [Logistic regression 3-class classifier](https://scikit-learn.org/stable/auto_examples/linear_model/plot_iris_logistic.html):

In [ ]:

_# TODO: Import all the necessary modules into this cell_

**Load the Iris dataset**

In [ ]:

_# TODO: Load the Iris dataset as X and Y arrays_

**Preprocess the data**

Preprocess the data using Scikit-learn methods, as you did in the Scikit-learn linear regression exercise:

- Randomly reorder the data.
- Normalise the data, if necessary.
- Divide the dataset into training and test subsets.

On this occasion, we will use K-fold cross-validation, as the dataset is very small (150 examples).

In [ ]:

_# TODO: Randomly reorder the data, normalise it only if necessary, and divide it into training and test subsets._

**Train an initial model**

- Train an initial model on the training subset without regularisation.
- Test the suitability of the model and retrain it if necessary.

The Scikit-learn function that you can use is [sklearn.linear\_model.LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) with an OvR scheme ("one-vs-rest", one class versus the rest).

Evaluate it on the test subset using its model.score():

In [ ]:

_# TODO: Train your model on the unregularised training subset and evaluate it on the test subset_

**Find the optimal regularisation using cross-validation**

- Train a model for each regularisation value to be considered.
- Train and evaluate them on a training subset fold using K-fold.
- Choose the optimal model and its regularisation.

The LogisticRegression method applies an L2 regularisation by default, although it uses the _C_ parameter which represents the inverse of _lambda_. Therefore, the lower the values, the greater the regularisation:

In [ ]:

_# TODO: Train a different model for each C on a different K-fold_

_# Use the values of lambda that we considered in previous exercises_

lambdas **=** [0., 1e-1, 1e1, 1e2, 1e3]

_# Calculate C for each lambda_

cs **=** [**...**]

_# Create 5 K-fold splits_

kf **=** [**...**]

_# Iterate over the 5 splits for your models and evaluate them on the generated CV subset_

log\_models **=** []

best\_model **=**** None**

**for** train, cv **in** kf **.** split(X):

_# Train a model on the training subset_

_# Remember to set the corresponding C parameter_

_# Evaluate it on the cv subset using its method score()_

_# Save the model with the best score for the best\_model variable and display the C of the best model._

c **=** [**...**]

print('L2 regularisation used:', c)

log\_models[**...**] **=** [**...**]

_# If the model is better than the best model so far..._

best\_model **=** [**...**]

print('L2 regularisation of the best model so far:', C)

**Finally, evaluate the model on the test subset**

- Display the coefficients and intercept of the best model.
- Evaluate the model on the test subset.
- Calculate the hits and misses on the test subset as in the linked example.

In [ ]:

_# TODO: Evaluate the best model on the initial test subset_

_# Display the coefficients and intercept of the best trained model_

print('Intercept coefficients of the trained model:')

print([...]) # _Display the intercept as the first coefficient_

_# Make predictions on the test subset_

y\_test\_pred **=** [**...**]

_# Calculate the average accuracy model evaluation metrics (its method score())_

mean\_accuracy **=** [**...**]

print('Mean accuracy: %.2f' **%** mean\_accuracy)

_# Calculate the hits and misses on the test subset and plot them graphically_

results **=** [**...**]

_# Plot them graphically_

plt **.** figure(1)

_# Fill in your code_

plt **.** show()
