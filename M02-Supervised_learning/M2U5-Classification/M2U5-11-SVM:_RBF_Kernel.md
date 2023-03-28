# **SVM: RBF Kernel**

M2U5 - Exercise 11

**What are we going to do?**

- We will generate a 2-class (binary) synthetic dataset
- We will preprocess the dataset
- We will train a classification model on the same dataset using SVM
- We will check its suitability
- Using validation, we will optimise the hyperparameters of our model
- We will evaluate our model

Remember to follow the instructions for the submission of assignments indicated in [Submission Instructions](https://github.com/Tokio-School/Machine-Learning/blob/main/Instrucciones%20entregas.md).

**Instructions**

Just as we had done for logistic regression classification using Scikit-learn, we are now going to use it to solve SVM classification problems.

Specifically, we are going to use its SVC classifier with the RBF ("Radial Basis Function") kernel. The Scikit-learn SVC model has several kernels available, and the Gaussian kernel in particular is not among them. However, the RBF kernel is closely related to it since it also starts from a "radial" classification, and in real projects it can actually be more efficient and perform better than the Gaussian kernel.

Therefore, instead of the Gaussian kernel we will use the RBF.

This SVM kernel has 2 parameters:

- _C._ The inverse of the regularisation parameter. For larger values of _C_, a smaller inter-class margin will be accepted if the decision function better classifies the training examples. Lower values of _C_ will attempt to increase the margin between classes, using a simpler decision function, possibly resulting in a lower accuracy.
- _Gamma_: Defines how far the influence of each example extends, or the inverse of the radius of influence of the examples selected by the model as "landmarks". Lower values will mean a more far-reaching influence and higher values a much closer influence.

We will optimise both parameters using cross-validation.

As a reference for this exercise, you can use these links from the documentation:

- [SVM: Classification](https://scikit-learn.org/stable/modules/svm.html#classification)
- [skelarn.svm.SVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)
- [RBF SVM parameters](https://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html)
- [SVM: Maximum margin separating hyperplane](https://scikit-learn.org/stable/auto_examples/svm/plot_separating_hyperplane.html)
- [Non-linear SVM](https://scikit-learn.org/stable/auto_examples/svm/plot_svm_nonlinear.html)

In [ ]:

_# TODO: Import all the necessary modules into this cell_

**Create a synthetic binary classification dataset**

Create a dataset for 2-class classification with [sklearn.datasets.make\_classification](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_classification.html).

Remember to use parameters that we can later modify, such as the number of examples, features, and classes, whether we want it to be unordered or not, a constant initial random state, number of clusters, etc.:

In [ ]:

_# TODO: Create a synthetic dataset for binary classification_

**Preprocess the data**

Preprocess the data:

- Randomly reorder the data.
- Normalise the data.
- Divide into training and test subsets (we will use CV by K-fold).

In [ ]:

_# TODO: Randomly reorder the data_

In [ ]:

_# TODO: Normalise the data if necessary_

In [ ]:

_# TODO: Divide into training, CV, and test subsets_

**Train an initial classification model using SVM**

To check the performance of our SVC classifier before optimising it by cross-validation, we will train an initial model on the training subset and validate it on the test subset.

Remember, use the [decision\_function\_shape](https://scikit-learn.org/stable/modules/svm.html#multi-class-classification) function to use the "one versus the rest" (OVR) scheme.

Use the default values for _C_ and _gamma_ so as not to influence their regularisation:

In [ ]:

_# TODO: Train an SVC model without modifying the regularisation parameters on the training subset._

In [ ]:

_# TODO: Evaluate the model with its model.score() on the test subset._

A very graphical way to better understand how SVMs work and to check the accuracy of your model is to represent the hyperplane that now separates the classes, whose margin with the classes we are trying to maximise.

To represent this, remember that you can follow the SVM example [: Maximum margin separating hyperplane](https://scikit-learn.org/stable/auto_examples/svm/plot_separating_hyperplane.html) and modifiy its code:

In [ ]:

_# TODO: Plot the separation hyperplane with the class margin_

**Optimise the regularization hyperparameters using cross-validation**

Once again we are going to use [sklearn.model\_selection.GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) to optimise our _C_ and _gamma_ hyperparameters using K-fold at the same time this time, and visually represent their possible values.

A very interesting example of this as we have mentioned is [RBF SVM parameters](https://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html).

Modify its code to optimise our model on our synthetic dataset, using K-fold to optimise _C_ and _gamma_. You can use the same logarithmic range of values for these hyperparameters or adapt it to this dataset.

_Note:_ Remember that we have previously preprocessed the dataset following our usual methods.

In [ ]:

_# TODO: Optimize the SVC hyperparameters_

_Note:_ Sometimes it is a good idea to split the code into different cells in order to be able to modify and re-execute them independently, especially when it takes time to execute:

In [ ]:

_# TODO: Plot the effect of the hyperparameters_

**Finally, evaluate the model on the test subset**

- Display the coefficients and intercept of the best model.
- Evaluate the best model on the initial test subset.
- Plot the class predictions to check hits, misses and the margin between classes on the new hyperplane.

To represent the predictions and the hyperplane margin between classes, you can reuse the code you used to evaluate the initial model:

In [ ]:

_# TODO: Evaluate the best model on the initial test subset_

In [ ]:

_# TODO: Plot the predictions, check the accuracy and the margin between classes_
