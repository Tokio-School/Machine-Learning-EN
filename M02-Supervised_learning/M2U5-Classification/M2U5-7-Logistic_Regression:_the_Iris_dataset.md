# **Logistic Regression: The Iris dataset**

M2U5 - Exercise 7

**What are we going to do?**

- We will download and analyse the Iris dataset
- We will preprocess the dataset
- We will train a classification model on the Iris dataset
- We will optimise our model using validation

Remember to follow the instructions for the submission of assignments indicated in [Submission Instructions](https://github.com/Tokio-School/Machine-Learning/blob/main/Instrucciones%20entregas.md).

**Instructions**

The Iris dataset is one of the best known and most widely used datasets in ML. It is commonly used as an example to explain model algorithms and is also widely used to compare various models with each other, based on their accuracy on the dataset.

You can learn more about this dataset on Wikipedia: [Iris flower dataset](https://en.wikipedia.org/wiki/Iris_flower_data_set)

You can find this model on Scikit-learn as [example dataset](https://scikit-learn.org/stable/datasets/toy_dataset.html#iris-dataset) and load it with the function [sklearn.datasets.load\_iris](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html).

In this exercise you will follow the same steps from the previous exercise to solve a 3-class classification model, in this case using a real dataset.

In [ ]:

_# TODO: Import all the necessary modules into this cell_

**Load the Iris dataset**

Load the dataset and analyse some of the examples to learn more about it.

To plot them graphically, you can follow this example: [The Iris Dataset](https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html)

In [ ]:

_# TODO: Load the Iris dataset as X and Y arrays_

_# Check to see what format Y is encoded in_

In [ ]:

_# TODO: Plot the distribution of examples on a 3D graph_

**Preprocess the data**

Preprocess the data following the usual steps:

- Randomly reorder the data.
- Normalise the data, only if necessary.
- Divide the dataset into training, validation, and test subsets.

_Note:_ Be careful when dividing them into subsets, _how many examples does the original dataset have?_

In [ ]:

_# TODO: Randomly reorder the data, normalise it only if necessary and divide it into training, validation, and test subsets_

**Implement the sigmoid activation function, the regularised cost function and the regularised gradient descent training function**

Copy code from previous exercises to implement these functions:

In [ ]:

_# TODO: Implement the sigmoid function_

In [ ]:

_# TODO: Implement the regularised cost and gradient descent functions_

**Train an initial model for each class**

To make sure your implementation works well with the dataset format, train a simple initial model for each class without regularisation.

Check to see if your implementation works well with the dataset format. If necessary, modify the dataset so that you can use it with your code.

Modify the training parameters and retrain if necessary:

In [ ]:

_# TODO: Train your models on the unregularised training subset_

**Find the optimal** _ **lambda** _ **hyperparameter using validation**

Once the initial model has been trained, and we have verified its performance, we are going to train a model for each of the 3 classes, with various values of _lambda_, as in previous exercises:

In [ ]:

_# TODO: Train a model for each different lambda value for each of the classes, and evaluate it on X\_val_

_# Use a logarithmic space between 10 and 10^3 with 5 elements with non-zero decimal values starting with a 1 or a 3_

_# By training more models, we can evaluate fewer lambda values to reduce the training time_

lambdas **=** [**...**]

In [ ]:

_# TODO: Plot the final error for each lambda value with one plot per class_

**Choosing the best model for each class**

Copy the code cells from the previous exercises and modify them as is necessary:

In [ ]:

_# TODO: Choose the optimal models and lambda values for each class on the validation subset_

**Evaluate the models on the test subset**

Once the models have been trained, evaluate them on the test subset in the same way as you did in previous exercises: for the whole test subset, make a prediction for each model in each class and choose the one with the best result after the sigmoid.

Calculate the residuals for these predictions and plot them graphically.

In this way, we are evaluating our model in a much more realistic environment, while also using a real dataset:

In [ ]:

_# TODO: Calculate the model error with the residuals on the test subset and plot them graphically_
