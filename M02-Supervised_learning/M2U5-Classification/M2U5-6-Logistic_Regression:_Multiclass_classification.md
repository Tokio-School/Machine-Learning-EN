# **Logistic Regression: Multiclass classification**

M2U5 - Exercise 6

**What are we going to do?**

- We will create a synthetic dataset for multiclass logistic regression
- We will preprocess the data
- We will train the model on the training subset and check its suitability
- We will find the optimal _lambda_ regularization parameter using CV
- We will make predictions about new examples

Remember to follow the instructions for the submission of assignments indicated in [Submission Instructions](https://github.com/Tokio-School/Machine-Learning/blob/main/Instrucciones%20entregas.md).

**Instructions**

Having implemented the full training of a regularised logistic regression model for binary (2 classes) classification, we will repeat the same example for multiclass classification (3+ classes).

In [ ]:

**import** time

**import** numpy **as** np

**from** matplotlib **import** pyplot **as** plt

**Create a synthetic dataset for multiclass logistic regression**

We will create a synthetic 3-class dataset for this complete implementation.

To do this, manually create a synthetic dataset for logistic regression with bias and error term (to have _Theta\_true_ available) with a slightly different code template than the one you used in the last exercise.

For the multiclass classification we will calculate Y in a different way: And it will have 2D (m x classes) dimensions, to represent all possible classes. We call this encoding of e.g. [0, 0, 1] for the 3/3 class "one-hot encoding":

- For each example and class, calculate _Y_ with the sigmoid with _Theta\_true_ and _X_.
- Transform the values of _Y_ to be 0 or 1 according to the max. value of the sigmoid of all the classes.
- Finally, transform the value of the class to 1 with a maximum value of the sigmoid, and the values of the other classes to 0, with a final ndarray for each example.

To introduce an error term, it runs through all Y values and changes the class of that example to a random class with a random error rate.

_NOTE:_ Investigate how a synthetic dataset for multiclass classification could be achieved using Scikit-learn methods.

**Implement the sigmoid activation function**

Copy your function from previous exercises:

In [ ]:

_# TODO: Implement the sigmoid function_

Create the synthetic dataset:

In [ ]:

_# TODO: Manually generate a synthetic dataset with a bias term and an error term_

_# Since we are going to train so many models, generate a "small" dataset in order to train them quickly_

_# If you need to, you can make it even smaller, or if you want more accuracy and a more realistic challenge, make it bigger_

m = 1000

n **=** 2

classes = 3

_# Generate a 2D m x n array with random values between -1 and 1_

_# Insert a bias term as a first column of 1's_

X **=** [**...**]

_# Generate a 2D theta array with (classes x n + 1) random values_

Theta\_true **=** [**...**]

_# Y shall have 2D dimensions of (m x classes)_

_# Calculate Y with the sigmoid and transform its values to 0 or 1 and then to one-hot encoding_

**for** i **in** range(m):

**for** c **in** range(classes):

sigmoid\_example **=** sigmoid([**...**])

_# Assign the only class corresponding to the example according to the max. value of the sigmoid_

Y[**...**] **=** [**...**]

_# To introduce an error term, go through all the Y values and change_

_# the class chosen from that example to another random class with a random % error_

_# Note: make sure that the other random class representing the error is different from the original one_

error **=** 0.15

**for** j **in** range(m):

_# If a random number is less than or equal to the error_

**if** [**...**]:

_# Assign a randomly selected class_

Y[**...**] **=** [**...**]

_# Check the values and dimensions of the vectors_

_print('Theta to be estimated:')_

print()

print('First 10 rows and 5 columns of X and Y:')

print()

print()

print('Dimensions of X and Y:')

print()

**Preprocess the data**

As we did for linear regression, we will preprocess the data completely, following the usual 3 steps:

- Randomly reorder the data.
- Normalise the data.
- Divide the dataset into training, validation, and test subsets.

**Randomly reorder the dataset**

Reorder the data in the _X_ and _Y_ dataset:

In [ ]:

_# TODO: Randomly reorder the dataset_

print('First 10 rows and 5 columns of X and Y:')

print()

print()

print('Reorder X and Y:')

_# Use an initial random state of 42, in order to maintain reproducibility_

X, Y **=** [**...**]

print('First 10 rows and 5 columns of X and Y:')

print()

print()

print('Dimensions of X and Y:')

print()

**Normalise the dataset**

Implement the normalisation function and normalize the dataset of _X_ examples:

In [ ]:

_# TODO: Normalise the dataset with the normalisation function_

_# Copy the normalisation function you used in the linear regression exercise_

**def** normalize(x, mu, std):

**pass**

_# Find the mean and standard deviation of the features of X (columns), except the first column (bias)_

mu **=** [**...**]

std **=** [**...**]

print('Original X:')

print(X)

print(X **.** shape)

print('Mean and standard deviation of the features:')

print()

print(mu **.** shape)

print(std)

print(std **.** shape)

print('Normalized X:')

X\_norm **=** np **.** copy(X)

X\_norm[...] = normalize(X[...], mu, std) _# Normalise only column 1 and the subsequent columns, not column 0_

print(X\_norm)

print(X\_norm **.** shape)

_Note:_ If you had modified your _normalize_ function to calculate and return the values of _mu_ and _std_, you can modify this cell to include your custom code.

**Divide the dataset into training, validation, and test subsets**

Divide the _X_ and _Y_ dataset into 3 subsets with the usual ratio, 60%/20%/20%.

If your number of examples is much higher or lower, you can always modify this ratio to another ratio such as 50/25/25 or 80/10/10.

In [ ]:

_# TODO: Divide the X and Y dataset into 3 subsets following the indicated ratios_

ratio **=** [60, 20, 20]

print('Ratio:\n', ratio, ratio[0] **+** ratio[1] **+** ratio[2])

r **=** [0, 0]

_# Tip: the round() function and the x.shape attribute may be useful to you_

r[0] **=** [**...**]

r[1] **=** [**...**]

print('Cutoff indices:\n', r)

_# Tip: the np.array\_split() function may be useful to you_

X\_train, X\_val, X\_test **=** [**...**]

Y\_train, Y\_val, Y\_test **=** [**...**]

print('Size of the subsets:')

print(X\_train **.** shape)

print(Y\_train **.** shape)

print(X\_val **.** shape)

print(Y\_val **.** shape)

print(X\_test **.** shape)

print(Y\_test **.** shape)

**Train an initial model for each class**

For multiclass classification, we must train a different model for each class. Therefore, if we have 3 classes we must train 3 different models.

Each model will only consider the values of the target variable relative to its class in a binary way, classifying examples as either belonging to its class or not belonging.

To do this, we will only provide you with the _Y_ values for that class or column. E.g., for Y = [[1, 0, 1], [0, 1, 0], [0, 0, 1]]:

- _Y_ for model 1: [1, 0, 0]
- _Y_ for model 2: [0, 1, 0]
- _Y_ for model 3: [0, 0, 1]

As we did in previous exercises, we will train initial models to check that our implementation is correct:

- Train an initial model without regularisation for each class.
- Plot the history of the cost function to check its evolution for each model.
- If necessary, modify any hyperparameters, such as the training rate, and retrain the models. You will use these hyperparameters in the following steps.

Copy the cells from previous exercises where you implemented the regularised cost function and gradient descent for logistic regression and the cell where you trained the model:

In [ ]:

_# TODO: Copy the cells with the cost and gradient descent functions for regularised classification_

In [ ]:

_# TODO: Train your models on the unregularised training subset_

_# Create an initial theta with a given value, which may or may not be the same for all the models._

theta\_ini **=** [**...**]

print('Theta initial:')

print(theta\_ini)

alpha **=** 1e-1

lambda\_ **=** 0.

e **=** 1e-3

iter\_ **=** 1e3

print('Hyperparameters used:')

print('Alpha:', alpha, 'Error máx.:', e, 'Nº iter', iter\_)

_# Initialise some variables to store the output of each model with the appropriate dimensions_

_# Caution: the models may require a different number of iterations before they converge_

_# Give j\_train a size to store up to the max. number of iterations, even if not all the elements are filled in_

j\_train\_ini **=** [**...**]

theta\_train **=** [**...**]

t **=** time **.** time()

**for** c **in** [**...**]: _# Iterate over the nº of classes_

print('\nModel for class nº:', c)

theta\_train = [...] _# Deep copy of theta\_ini to remain unchanged_

t\_model **=** time **.** time()

j\_train\_ini[**...**], theta\_train[**...**] **=** regularized\_logistic\_gradient\_descent([**...**])

print('Training time for model (s):', time.time() - t\_model)

print('Total training time (s):', time.time() - t)

print('\nFinal cost of the model for each class:')

print()

print('\nFinal theta of the model for each class:')

print()

In [ ]:

_# TODO: Plot the evolution of the cost function vs. the number of iterations for each model_

plt **.** figure(1)

plt.title('Cost function for each class')

**for** c **in** range(classes):

plt **.** subplot(classes, 1, c **+** 1)

plt **.** xlabel('Iterations')

plt **.** ylabel('Class cost {}' **.** format(c))

plt **.** plot(j\_train\_ini[**...**])

plt **.** show()

**Test the suitability of the models**

Check the accuracy of your models and modify the parameters to retrain them if necessary.

Remember that if your dataset is "too accurate" you can go back to the original cell and enter a higher error term.

Due to the complexity of multiclass classification, we will not ask you on this occasion to check whether the models may be suffering from deviation or overfitting.

**Find the optimal** _ **lambda** _ **hyperparameter using validation**

As we have done in previous exercises, we will optimise our regularisation parameter by validation for each of the classes and models.

Now, in order to find the optimal _lambda,_ we will train a different model on the training subset for each _lambda_ value to be considered and check its accuracy on the validation subset.

Again, we will plot the error of each model vs. the _lambda_ value used and implement a code that automatically chooses the most optimal model for each class.

Remember to train all your models under equal conditions, with the same hyperparameters.

Therefore, you must now modify the preceding cell's code so that you do not train one model like before, but rather one model per class and for each of the _lambda_ values to be considered:

In [ ]:

_# TODO: Train a model on X\_train for each different lambda value and evaluate it on X\_val_

_# Use a logarithmic space between 10 and 10^3 with 5 elements with non-zero decimal values starting with a 1 or a 3_

_# By training more models, we can evaluate fewer lambda values to reduce training time_

lambdas **=** [**...**]

_# Complete the code to train a different model for each class and value of lambda on X\_train_

_# Store your thetas and final costs_

_# Afterwards, evaluate its total cost on the validation subset_

_# Store this information in the following arrays_

_# Careful with its essential dimensions_

j\_train **=** [**...**]

j\_val **=** [**...**]

theta\_val **=** [**...**]

In [ ]:

_# TODO: Plot the final error for each lambda value with one plot per class_

plt **.** figure()

_# Fill in your code_

**for** c **in** range(classes):

plt **.** subplot(classes, 1, c **+** 1)

plt **.** title('Class:', c)

plt **.** xlabel('Lambda')

plt **.** ylabel('Final cost')

plt **.** plot(j\_train[**...**])

plt **.** plot(j\_val[**...**])

plt **.** show()

**Choosing the best model for each class**

_Copy the code from previous exercises and modify it to choose the most accurate model on the validation subset for each class:_

In [ ]:

_# TODO: Choose the optimal models and lambda values, with the lowest error on the validation subset_

_# Iterate over all the combinations of theta and lambda and choose the lowest cost models on the validation subset for each class_

j\_final **=** [**...**]

theta\_final **=** [**...**]

lambda\_final **=** [**...**]

**Evaluate the models on the test subset.**

Finally, we will evaluate the model of each class on a subset of data that we have not used for training nor for choosing any hyperparameters.

Therefore, we will calculate the total cost or error on the test subset and graphically check the residuals of the model on it.

Remember to use only the _Y_ columns that each model would "see", as it classifies examples according to whether they belong to its class or not.

In [ ]:

_# TODO: Calculate the error of the models on the test subset using the cost function_

_# Use the theta and lambda of the specific model of the class corresponding to that example_

j\_test **=** [**...**]

In [ ]:

_# TODO: Calculate the predictions of the models on the test subset, calculate the residuals and plot them_

_# Remember to use the sigmoid function to transform the predictions and choose the class according to the maximum value of the sigmoid_

Y\_test\_pred **=** [**...**]

residuals **=** [**...**]

plt **.** figure(4)

_# Fill in your code_

plt **.** show()

**Make predictions about new future examples**

With our model trained, optimised, and evaluated, all that remains is to put it to work by making predictions with new examples.

To do this, we will:

- Generate a new example, which follows the same pattern as the original dataset.
- Normalise its features before making predictions about them.
- Generate a prediction for this new example for each of the classes, for each of the 3 models.
- Choose the class with the highest _Y_ value after the sigmoid as the final class, even though several models predicted Y ≥ 0.0; Y = 1.

In [ ]:

_# TODO: Generate a new example following the original pattern, with a bias term_

X\_pred **=** [**...**]

_# For comparison, before normalising the data, use Theta\_true to see what the actual associated class would be_

Y\_true **=** [**...**]

_# Normalise its features (except for the bias term) with the means and standard deviations of the training subset_

X\_pred **=** [**...**]

_# Generate a prediction for this new example for each model using the sigmoid_

Y\_pred **=** [**...**]

_# Choose the highest value after the sigmoid as the final class and transform it to a one-hot encoding vector of 0 and 1._

Y\_pred **=** [**...**]

_# Compare the actual class associated with this new example and the predicted class_

print('Actual class of the new example and predicted class:')

print(Y\_true)

print(y\_pred)
