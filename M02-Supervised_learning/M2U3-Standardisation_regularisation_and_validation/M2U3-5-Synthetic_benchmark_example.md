# **Linear Regression: Synthetic Benchmark Example**

M2U3 - Exercise 5

**What are we going to do?**

- We will create a reference notebook with all the steps to train a multivariate linear regression model
- We will clean up the code of any leftover comments or explanations
- We will modify the original dataset to see how it affects the trained model

Remember to follow the instructions for the submission of assignments indicated in [Submission Instructions](https://github.com/Tokio-School/Machine-Learning/blob/main/Instrucciones%20entregas.md).

This exercise will be different. The last notebook was quite long and contained a lot of additional information and explanations.

We will create a reference notebook with all the steps to train a multivariate linear regression model. We will also use this notebook as a reference to train other, non-linear regression models, and optimise their hyperparameters with a validation subset.

We also want you to take the opportunity to clean up your code cells, removing any instruction comments, hints, etc., so that you can use these cells or notebook directly in the future.

Therefore, your task in this exercise will be simple: copy the code cells from previous exercises, clean them up and leave them ready as a reference for training linear regression models.

_Note:_ You can modify and add any markdown cell to add some explanation or summary to help you understand the steps to follow.

**Create a synthetic dataset for linear regression**

- Create it manually or with Scikit-learn's specific methods.
- Add a modifiable bias and error term.

**Preprocess the data**

- Randomly reorder the data.
- Normalise the data.
- Divide the dataset into training, validation, and test subsets.

**Train an initial model**

- Train an initial model on the training subset without regularisation.
- Plot the history of the cost function to check its evolution.
- Test the suitability of the model.
- Retrain the model by varying the hyperparameters if necessary.
- Check if there is any deviation or overfitting.

**Find the optimal** _ **lambda** _ **on the validation subset**

- Train a model for each _lambda_ value to be considered
- Plot the final error over the training and validation subset of each model/_lambda_.
- Choose the optimal model and _lambda_ value

**Finally, evaluate the model on the test subset**

- Calculate the cost of the model on the test subset using the corresponding _theta_ and _lambda_
- Display a summary of your evaluation metric(s): RMSE, R^2, etc.
- Calculate the residuals on the test subset and plot them.

**Make predictions about new examples**

- Generate a new example, following the same pattern as the original dataset.
- Normalise its features.
- Generate a prediction for this new example.

**Additional challenge**

Imagine that in a real-world event that we want to model, we have not collected all the features that really affect the event. E.g., if it was affected by 25 features and we only estimated or were able to capture 12 of them.

How might this affect the model?

_Can you create a synthetic dataset with a larger number of features that determine Y than the number of features used to train the model, and test it?_
