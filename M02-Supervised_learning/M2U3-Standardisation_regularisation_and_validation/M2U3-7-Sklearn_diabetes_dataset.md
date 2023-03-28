# **Linear Regression with Scikit-learn: The Diabetes dataset**

M2U3 - Exercise 7

**What are we going to do?**

- We will analyse the Scikit-learn diabetes sample dataset
- We will train a multivariate linear regression model on the dataset
- Examine the Lasso linear regression model
- We will compare the CV included in LassoCV with a manual CV

Remember to follow the instructions for the submission of assignments indicated in [Submission Instructions](https://github.com/Tokio-School/Machine-Learning/blob/main/Instrucciones%20entregas.md).

# **Linear Regression: Scikit-learn Diabetes dataset**

**What are we going to do?**

- We will analyse the Scikit-learn diabetes sample dataset
- We will train a multivariate linear regression model on the dataset
- We will examine the Lasso linear regression model
- We will compare the CV included in LassoCV with a manual CV

For this exercise you can use the following references, among others:

- [Diabetes dataset](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset)
- [sklearn.datasets.load\_diabetes](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html#sklearn.datasets.load_diabetes)
- [Cross-validation on Diabetes dataset exercise](https://scikit-learn.org/stable/auto_examples/exercises/plot_cv_diabetes.html)
- [sklearn.model\_selection.GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)

In [ ]:

_# TODO: Import all the necessary modules into this cell_

**Load the Diabetes dataset**

Before starting to work with a new dataset, it is worth reviewing its features and displaying some examples on screen to get an idea of what data it contains, in which format, etc:

_NOTE:_ If you have experience using Pandas, you can use the dataframe.describe() function

In [ ]:

_# TODO: Load the Diabetes dataset_

_# First, load it as a Bunch object, analyse its features and show some examples_

_# Then modify your code and load it as a tuple (X, Y) or as a Pandas dataset, if you prefer_

**Preprocess the data**

- Randomly reorder the data
- Normalise the data
- Divide the dataset into training and test subsets

_Note:_ Before normalising the data from a new dataset, check whether it is necessary and ensure it has not already been normalised.

_Hint_: [Diabetes dataset](https://scikit-learn.org/stable/datasets/toy_dataset.html#:~:text=blood%20sugar%20level-,Note,-%3A%20Each%20of%20these)

In [ ]:

_# TODO: Randomly reorder and normalise the data and split the dataset into 2 subsets, as needed_

**Train an initial model**

- Train an initial model on the training subset without regularisation.
- Test the suitability of the model.
- Check if there is any deviation or overfitting on the test subset(on this occasion).

If so, revert to using a simpler linear regression model, such as LinearRegression:

In [ ]:

_# TODO: Train a simpler linear regression model on the training subset without regularisation._

In [ ]:

_# TODO: Test the suitability of the model by evaluating it on the test set using various metrics_

In [ ]:

_# TODO: Check if the evaluation on both subsets is similar with the RMSE_

**Train the model with CV**

- Train a model for each regularisation value to be considered.
- Train and evaluate them on the training subset using K-fold.
- Choose the optimal model and its regularisation.

Train the model using the Lasso [algorithm](https://scikit-learn.org/stable/modules/linear_model.html#lasso):

_NOTE:_ You can refer to the notebook for the referenced exercise.

_QUESTION_: What kind of regularisation does this algorithm implement?

In [ ]:

_# TODO: Train a different model for each alpha on a different K-fold, evaluate them and select_

_# the most accurate model using GridSearchCV_

_Optional bonus:_ Do you have the courage to plot the same graphical representation of the evaluation of each model for multiple _alphas?_

**Finally, evaluate the model on the test subset**

- Display the coefficients (bias and intercept) of the best model.
- Evaluate the best model on the initial test subset.
- Calculate the residuals for the test subset and plot them.

In [ ]:

_# TODO: Evaluate the best model on the initial test subset and calculate its residuals_

_ **Bonus:** _ **Compare the results of our CV with those of the LassoCV**

Several algorithms use an additional implementation that performs a prior CV optimisation internally, as is the case with [LassoCV](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoCV.html#sklearn.linear_model.LassoCV).

However, in many cases we cannot be completely confident that they have chosen the best _alpha_. Therefore, it may be a good idea to compare the results of LassoCV with those of our Lasso model with manual CV.

_Do you have the courage to implement it yourself?_

In [ ]:

_# TODO: Compare the results of our manually trained CV model with those of LassoCV._

_QUESTIONS:_

- How much can you rely on LassoCV's _alpha_ selection?
- Are the _alphas_ chosen by LassoCV and their accuracies similar to those of our model?
