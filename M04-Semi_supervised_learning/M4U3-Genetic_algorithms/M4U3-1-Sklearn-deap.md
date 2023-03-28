# **Genetic algorithms: Sklearn-deap**

M4U3 - Exercise 1

**What are we going to do?**

- We will optimize training hyperparameters to validate a model using genetic algorithms
- We will use the DEAP library via SKlearn-deap

**Instructions**

Genetic algorithms are very useful for a wide variety of cases, where we want to optimise our system or function in an evolutionary way.

However, they are particularly difficult to use effectively, as they add a large number of parameters that are not always easy to map to real system parameters: phenotypes, genotypes, mutational probabilities, mating and survival, etc.

Moreover, they are not always particularly efficient at finding the most optimal solution and may fall into local minima.

In this exercise we are going to give a very simple example of using, instead of Scikit-learn, validation methods like [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) and [RandomizedSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html).

For this we will use the SKlearn-deap [library](https://github.com/rsteca/sklearn-deap) to train a SVM classification model using cross-validation (K-fold).

Follow the instructions in the following cells, building on the code from past exercises where possible and in this [example](https://github.com/rsteca/sklearn-deap/blob/master/test.ipynb) using SKlearn-deap:

In [ ]:

_# TODO: Use this cell to import all the necessary libraries_

In [ ]:

_# TODO: Create a synthetic dataset for classification with a significant error parameter_

In [ ]:

_# TODO: Preprocess the data: normalise it, split it into 2 training and test subsets and randomly reorder it_

In [ ]:

_# TODO: Train an SVM model with cross-validation using genetic algorithms to select the best combination of_

_# kernel and hyperparameters_

In [ ]:

_# TODO: Evaluate the F1-score of the model on the test subset._

_BONUS_: As you can see on the front page of the SKlearn-deap repository, it can also be used to maximise or minimise custom functions.

_Could you use SKlearn-deap to minimize your manual implementation of the cost function for logistic regression?_

_NOTE:_ Try to use an error-free dataset with a low number of _n_-dimensions and no regularization.

In [ ]:

_# BONUS TODO: Create a synthetic dataset for classification without an error parameter_

In [ ]:

_# BONUS TODO: Copy your cost function for regularized logistic regression_

In [ ]:

_# BONUS TODO: Optimize Theta by minimising your cost function using SKlearn-deap_
