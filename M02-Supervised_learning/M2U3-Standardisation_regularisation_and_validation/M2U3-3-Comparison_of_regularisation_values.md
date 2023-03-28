# **Linear Regression: Comparison of regularisation values**

M2U3 - Exercise 3

**What are we going to do?**

- Create a synthetic dataset for multivariate linear regression with a random error term
- We will train 3 different linear regression models on this dataset with different _lambda_ values
- We will compare the effect of the _lambda_ value on the model, its accuracy, and its residuals

Remember to follow the instructions for the submission of assignments indicated in [Submission Instructions](https://github.com/Tokio-School/Machine-Learning/blob/main/Instrucciones%20entregas.md).

**Create a synthetic dataset with error term for training and final testing**

We will start, as usual, by creating a synthetic dataset for linear regression, with bias and error terms, either manually or with Scikit-learn methods.

This time we are going to create 2 datasets, one for training and one for final test, following the same pattern but with different sizes. We will train the models with the first dataset and then check with the second dataset how they would behave on data that they have not "seen" previously in the training process, which are completely new to them.

In [ ]:

_# TODO: Generate a synthetic dataset manually, with bias term and error term_

m = 100

n = 1

X\_train **=** [**...**]

X\_test **=** [**...**] _# The size of the test dataset should be 25% of the original_

Theta\_true **=** [**...**]

error **=** 0.35

Y\_train **=** [**...**]

Y\_test **=** [**...**]

_# Check the values and dimensions of the vectors_

print('Theta and its dimensions to be estimated:')

print()

print()

_# Check X\_train, X\_test, Y\_train and Y\_test_

print('First 10 rows and 5 columns of X and Y:')

print()

print()

print()

print()

print('Dimensions of X and Y:')

print()

print()

**Train 3 different models with different** _ **lambda** _ **values**

We will train 3 different models on this dataset with different _lambda_ values.

To do this, start by copying your cells with the code that implements the regularised cost function and gradient descent:

In [ ]:

_# TODO: Copy here the cells or the code to implement 2 functions with regularised cost function and gradient_

_descent_

Let's train the models. To do this, remember that with Jupyter you can simply modify the code cells and the variables will remain in the Jupyter kernel memory.

Therefore, you can e.g., modify the name of the following variables, changing "1" to "2" and "3", and simply re-execute the cell to store the results of the 3 models, while the variables of the previous models are still available.

If you run into any difficulties, you can also copy the code cell several times and have 3 cells to train 3 models with different variable names.

In [ ]:

_# TODO: Test your implementation by training a model on the previously created synthetic dataset._

_# Create an initial theta with a given constant value (not randomly this time)._

theta\_ini **=** [**...**]

print('Theta initial:')

print(theta\_ini)

alpha **=** 1e-1

lambda\_ **=** [1e-3, 1e-1, 1e1] _# We use 3 different values_

e **=** 1e-3

iter\_ **=** 1e3 _# Check that your function can support float values or modify it_

print('Hyperparameters used:')

print('Alpha:', alpha, 'Error máx.:', e, 'Nº iter', iter\_)

t **=** time **.** time()

_# Use lambda\_[i], within the range [0, 1, 2] for each model_

j\_hist\_1, theta\_final\_1 **=** gradient\_descent([**...**])

print('Training time (s):', time **.** time() **-** t)

_# TODO: complete_

print('\nLast 10 values of the cost function')

print(j\_hist\_1[**...**])

print('\Final cost:')

print(j\_hist\_1[**...**])

print('\nTheta final:')

print(theta\_final\_1)

print('True values of Theta and difference with trained values:')

print(Theta\_true)

print(theta\_final\_1 **-** Theta\_true)

**Graphically check the effect of** _ **lambda** _ **on the models**

Now let's check the 3 models against each other.

Let's start by checking the final cost, a representation of their accuracy:

In [ ]:

_# TODO: Show the final cost of the 3 models:_

print('Final cost of the 3 models:')

print(j\_hist\_1[**...**])

print(j\_hist\_2[**...**])

print(j\_hist\_3[**...**])

_# Visually represent the cost vs. lambda values with a line and dot plot_

plt **.** plot([**...**])

_How does a higher_ lambda _value affect the final cost in this dataset?_

Let's plot the training and test datasets, to check that they follow a similar pattern:

In [ ]:

_# TODO: Plot X\_train vs Y\_train, and X\_test vs Y\_test graphically._

plt **.** figure(1)

plt **.** title([**...**])

plt **.** xlabel([**...**])

plt **.** ylabel([**...**])

_# Remember to use different colours_

plt **.** scatter([**...**])

plt **.** scatter([**...**])

_# Create a legend for the different series and their colours_

plt **.** show()

We will now check the predictions of each model on the training dataset, to see how well the line fits the training values in each case:

In [ ]:

_# TODO: Calculate the predictions for each model on X\_train_

Y\_train\_pred1 **=** [**...**]

Y\_train\_pred2 **=** [**...**]

Y\_train\_pred3 **=** [**...**]

In [ ]:

_# TODO: For each model, graphically represent its predictions about X\_train_

_# If you get an error with other notebook charts, use the bottom line of plt.figure() or comment it out_

plt **.** figure(2)

fig, (ax1, ax2, ax3) **=** plt **.** subplots(3, sharex **=True** , sharey **=True** )

fig **.** suptitle([**...**])

_# Use different colours for each model_

ax1 **.** plot()

ax1 **.** scatter()

ax2 **.** plot()

ax2 **.** scatter()

ax3 **.** plot()

ax3 **.** scatter()

Since the training dataset has an error term, there may be significant differences between the data in the training dataset and the test dataset. You can play with various values of this term to increase or decrease the difference.

Let's check what happens to the predictions when we plot them on the test dataset, on data that the models have not seen before:

In [ ]:

_# TODO: Calculate predictions for each model on X\_test_

Y\_test\_pred1 **=** [**...**]

Y\_test\_pred2 **=** [**...**]

Y\_test\_pred3 **=** [**...**]

In [ ]:

_# TODO: For each model, graphically represent its predictions about X\_test._

_# If you get an error with other notebook charts, use the bottom line of plt.figure() or comment it out_

plt **.** figure(3)

fig, (ax1, ax2, ax3) **=** plt **.** subplots(3, sharex **=True** , sharey **=True** )

fig **.** suptitle([**...**])

_# Use different colours for each model_

ax1 **.** plot()

ax1 **.** scatter()

ax2 **.** plot()

ax2 **.** scatter()

ax3 **.** plot()

ax3 **.** scatter()

What happens? In some cases, depending on the parameters used, it may be more or less easy to discern it.

When the model has a low or zero _lambda_ regulation factor, it fits too closely to the data on which it is trained, achieving a very tight curve and maximum accuracy... only on that particular dataset.

However, in real life, data on which we have not trained the model may subsequently arrive that has some small variation on the original data.

In such situations we prefer a higher _lambda_ value, which allows us to have a higher accuracy for the new data, even if we lose some accuracy for the training dataset data.

We are therefore looking for a model that can "generalise" and be able to make good predictions about new data, rather than one that simply " memorizes" the results it has already seen.

We can therefore think of regularisation as a student who has the exam questions before sitting the exam:

- If he then gets those questions, he will have a very high mark (or accuracy), as he has already "seen" the questions beforehand.
- Then, if the questions are different, he may still have a high score, depending on how similar they are.
- However, if the questions are totally different, he will get a very low mark, because it is not that he had thoroughly studied the subject, but that his marks were high just because he knew the results beforehand.

**Check the residuals on the final test subset**

Plot the residuals for the 3 models graphically. That way you will be able to compare your 3 models on the 2 datasets.

Calculate the residuals for both the training and testing datasets. You can do this with different cells to be able to appreciate their differences simultaneously.

_Tips:_

- Be careful with the scales of the X and Y axes when making comparisons.
- To be able to see them at the same time, you can create 3 horizontal subplots, instead of vertical ones, with "plt.subplots(1, 3)".
- # Use different colours for each of the 3 models.

If you do not clearly see such effects on your datasets, you can try modifying the initial values:

- With a larger number of examples, so that the models can be more accurate.
- With a larger error term, so that there is more difference or variation between examples.
- With a smaller size test dataset over the training, so that there are more differences between the two datasets (having more data, the values can be more smoothed).
- Etc.
