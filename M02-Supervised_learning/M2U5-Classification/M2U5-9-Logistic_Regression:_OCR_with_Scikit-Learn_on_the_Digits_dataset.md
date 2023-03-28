# **Logistic Regression: OCR with Scikit-Learn on the Digits dataset**

M2U5 - Exercise 9

**What are we going to do?**

- We will download the handwritten digits dataset to classify it using OCR (optical character recognition)
- We will preprocess the dataset using Scikit-learn methods
- We will train a multiclass classification model using Scikit-learn

Remember to follow the instructions for the submission of assignments indicated in [Submission Instructions](https://github.com/Tokio-School/Machine-Learning/blob/main/Instrucciones%20entregas.md).

**Instructions**

- xxx

# **Logistic Regression: Scikit-learn on the Digits dataset using OCR**

**What are we going to do?**

- We will download the handwritten digits dataset to classify it using OCR (optical character recognition).
- We will preprocess the dataset using Scikit-learn methods.
- We will train a multiclass classification model using Scikit-learn.

OCR is a set of techniques related to machine-learning and deep-learning or neural networks that attempts to visually recognise handwritten characters.

As the character set is relatively small (10 classes), it is a model that we can sometimes simply solve using logistic classification or SVM.

- You can find the features of the dataset here: [Optical recognition of handwritten digits dataset](https://scikit-learn.org/stable/datasets/toy_dataset.html#digits-dataset)
- You can load it with this function: [sklearn.datasets.load\_digits](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html)
- You can use this notebook as a reference: [Recognising hand-written digits](https://scikit-learn.org/stable/auto_examples/classification/plot_digits_classification.html)

Repeat the steps of the previous exercise to train an OCR ML model on this dataset with Scikit learn's [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) function:

In [ ]:

_# TODO: Import all the necessary modules into this cell_

**Load the Digits dataset**

Before starting to work with the dataset, graph some of the examples and their associated classes or digits:

In [ ]:

_# TODO: Load the Digits dataset as X and Y arrays representing some of the examples_

**Preprocess the data**

Preprocess the data using Scikit-learn methods, as you did in the Scikit-learn linear regression exercise:

- Randomly reorder the data.
- Normalise the data, if necessary.
- Divide the dataset into training and test subsets.

On this occasion, we will use K-fold cross-validation, as the dataset is very small (150 examples).

In [ ]:

_# TODO: Randomly reorder the data, normalize it only if necessary, and divide it into training and test subsets._

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

The LogisticRegression function applies an L2 regularisation by default, although it uses the _C_ parameter which represents the inverse of _lambda:_

In [ ]:

_# TODO: Train a different model for each C on a different K-fold_

**Finally, evaluate the model on the test subset**

- Display the coefficients and intercept of the best model.
- Evaluate the best model on the initial test subset.
- Calculate the hits and misses on the test subset and plot them graphically.

As this dataset is very visual, try to also show the examples where the model has failed visually, and consider whether you would be able to recognise that number.

_Sometimes even a human would have trouble deciphering it based on the handwriting of the writer 8)._

In [ ]:

_# TODO: Evaluate the best model on the initial test subset and plot its misses graphically._
