# **Linear Regression: Training Hyperparameters**

M2U2 - Exercise 6

**What are we going to do?**

- We will test the effect of different hyperparameters on the training of a model

Remember to follow the instructions for the submission of assignments indicated in [Submission Instructions](https://github.com/Tokio-School/Machine-Learning/blob/main/Instrucciones%20entregas.md).

**Effect of the hyperparameters**

In this lab we are not going to introduce any new concepts or implement any code that we have not implemented in previous exercises.

The aim of this exercise is to have an opportunity to test how different hyperparameters and other settings affect our training process.

To do this, you are simply going to copy your code cells from the previous exercise ([Linear Regression): Example of a synthetic dataset](https://github.com/Tokio-School/Machine-Learning/blob/main/M02-Aprendizaje_supervisado/M2U2-Optimizaci%C3%B3n_por_descenso_de_gradiente/M2U2-5-Dataset_sint%C3%A9tico.ipynb), you will modify the hyperparameters many times and test their effect on the training, and validate your implementation again, which you will then use in multiple subsequent exercises.

By the way, _do you know the JupyterLab hotkeys? They will be very useful during the course:_ [https://jupyterlab.readthedocs.io/en/stable/user/interface.html#keyboard-shortcuts](https://jupyterlab.readthedocs.io/en/stable/user/interface.html#keyboard-shortcuts)

In [1]:

**import** time

**import** numpy **as** np

**from** matplotlib **import** pyplot **as** plt

**Creation of a synthetic dataset**

Copy the cell from the previous exercise to create a synthetic dataset, with a bias term and an error term. Give the error term a value other than 0.

In [2]:

_# TODO: Copy the corresponding cell code here, or copy the whole cell and delete it._

**Training the model**

Copy the cell with the cost and gradient descent functions and the cell that trains the model using gradient descent.

**Check the training of the model**

Copy the cell that plots the historical cost function of your model.

**Model evaluation**

Copy the cell that calculates and graphs the residuals of your model.

Add to that cell the predicted Y calculation _(Y\_pred)_ from the previous cell, as without that variable your code will not work correctly.

**Modify the hyperparameters and check their effect.**

Now proceed to modify the different hyperparameters one by one and check their effect. The training hyperparameters and other configuration parameters that we are going to modify are the following:

1. _m_ and _n_
2. The dataset error term, _error_
3. The training rate, _alpha_
4. The convergence parameter, _e_
5. The maximum number of iterations, _iter\__

We want you to be as autonomous as possible in this task, to take this time to experiment, to discover the behaviour of these parameters for yourself, to learn how to modify them in the future, when we need to modify them to optimise the training of more complex models.

So, _why not take this opportunity to enjoy and discover for yourself what happens when we modify them, both individually and jointly?_

When you are ready to continue, you can move on to the last section

**Hyperparameter modification: questions and conclusions**

We are including some simple questions for you to answer as an evaluation exercise based on your assessment in the previous topic. You can modify this markdown cell and add your answers to it.

1. _What happens when we increase the number of examples?_
2. _How does it affect the training time?_
3. _How does it affect the accuracy or final cost of the model_?
4. _What happens when we increase the number of features? How does this impact the training time and the accuracy of the model?_
5. _How does the error term affect your training? Its accuracy, the number of iterations until convergence..._
6. _Does modifying the maximum number of iterations affect the training? Does it have any impact on the training time, the final accuracy, etc.?_
7. _Is there a limit to the maximum number of iterations, or can we increase this number to infinity to achieve e.g., improved accuracy?_
8. _Is the maximum number of iterations related to the training convergence parameter?_
9. _Is the convergence parameter related to the final cost of the model?_
10. _How does the training rate affect the model? Its speed, its accuracy?_
11. _Can we increase the training rate to infinity, or is there a limit beyond which it stops working for both maximum and minimum values?_

Curiosities: Let's test your current working environment. This will give us an idea of its robustness, but also a baseline for you to remember in the future and to know its limits:

1. _What is the maximum number of examples and features supported by the resources of your current working environment?_ Run several tests, and when you reach the limit of features, modify the number of examples. Obtain about 3 points from combinations of both values from which your environment returns a resource error.
2. Get an idea of the training time needed for a typical model in your environment for a basic algorithm like linear regression. With a 15% error term, a limit on iterations, a learning rate, and a sufficient convergence parameter, write down the time it takes to train the model on your computer. Try various values for the number of examples and features, e.g., a "small", "medium", and "large" size dataset.

In []:

_# TODO: Visually represent the training times for the 3 dataset sizes on a line graph._

_Finally, are there any additional findings that you have discovered that are not covered in the questions above? We look forward to hearing from you!_
