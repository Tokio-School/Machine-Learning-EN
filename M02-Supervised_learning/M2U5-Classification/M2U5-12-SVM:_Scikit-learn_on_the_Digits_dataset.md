# **SVM: Scikit-learn on the Digits dataset**

M2U5 - Exercise 12

**What are we going to do?**

- We will download the Digits dataset to classify it using OCR
- We will preprocess the dataset using Scikit-learn methods
- We will train a multiclass classification model using SVM
- We will evaluate the accuracy of the model and represent it graphically

Remember to follow the instructions for the submission of assignments indicated in [Submission Instructions](https://github.com/Tokio-School/Machine-Learning/blob/main/Instrucciones%20entregas.md).

**Instructions**

In a previous exercise, we solved the problem of OCR handwritten digit classification by logistic regression using Scikit-learn.

This application and dataset can be complex, so we expect an SVM classification model to be more accurate than a linear logistic regression one.

Repeat the steps to train an SVM model on this dataset, optimise it by CV with K-fold and finally, evaluate it.

References:

- You can find the features of the dataset here: [Optical recognition of handwritten digits dataset](https://scikit-learn.org/stable/datasets/toy_dataset.html#digits-dataset)
- Loading function: [sklearn.datasets.load\_digits](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html)
- You can use this notebook as a reference: [Recognising hand-written digits](https://scikit-learn.org/stable/auto_examples/classification/plot_digits_classification.html)

In [ ]:

_# TODO: Import all the necessary modules into this cell_

**Load the Digits dataset**

Plot some of the examples and their associated classes or digits:

In [ ]:

_# TODO: Load the Digits dataset as X and Y arrays representing some of the examples_

**Preprocess the data**

Preprocess the data using Scikit-learn methods, as you did in the Scikit-learn logistic regression exercise:

- Randomly reorder the data.
- Normalise the data, if necessary.
- Divide the dataset into training and test subsets.

On this occasion, we will once again use K-fold for cross-validation.

In [ ]:

_# TODO: Randomly reorder the data_

In [ ]:

_# TODO: Normalise the data, only if necessary_

In [ ]:

_# TODO: Divide the dataset into training and test subsets_

**Train an initial classification model using SVM**

To check the performance of our SVC classifier, we will train an initial model on the training subset and validate it on the test subset.

Remember, use the [decision\_function\_shape](https://scikit-learn.org/stable/modules/svm.html#multi-class-classification) function to use the "one versus the rest" (OVR) scheme.

Use the default values for _C_ and _gamma_ so as not to influence their regularisation:

In [ ]:

_# TODO: Train an SVC model without modifying the regularisation parameters on the training subset._

To evaluate the model, we can use a confusion matrix. In this matrix we represent, for each of the classes (10 in this case), how many examples have been predicted correctly, how many have been misclassified, and which classes we had predicted for them.

You can represent the confusion matrix using the [sklearn.metrics.plot\_confusion\_matrix](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.plot_confusion_matrix.html) function that you can find in this example: [Recognising hand-written digits](https://scikit-learn.org/stable/auto_examples/classification/plot_digits_classification.html)

NOTE: Do you have the courage to use a more appropriate colour scale for this case?

In [ ]:

_# TODO: Evaluate the model with its model.score() on the test subset._

In [ ]:

_# TODO: Plot the confusion matrix for the model_
