# **Linear Regression: Home appraiser example**

M2U2 - Exercise 7

**What are we going to do?**

- We will train a multivariate linear regression model
- We will create a synthetic dataset following a real data schema

Remember to follow the instructions for the submission of assignments indicated in [Submission Instructions](https://github.com/Tokio-School/Machine-Learning/blob/main/Instrucciones%20entregas.md).

In [ ]:

**import** time

**import** numpy **as** np

**from** matplotlib **import** pyplot **as** plt

**Synthetic house valuation dataset**

This time we are going to explore how to create a synthetic dataset that follows the structure we want to simulate a real dataset with full flexibility.

In this case, we are going to use a real estate dataset as an example, with the objective of training a housing dataset with the following features:

Target or dependent variable:

- Sale price (int)

Features, explanatory or independent variables:

- Useful surface area (int)
- Location (int, representing the neighbourhood as an ordinal category)
- Type of dwelling (int, representing flat, detached house, semi-detached house, penthouse, etc. as an ordinal category)
- Nº of rooms (int)
- Nº of bathrooms (int)
- Garage (int, 0/1 representing whether it has one, or not)
- Surface area of common areas (int)
- Year of construction (int)

Our model will attempt to approximate a multivariate linear function that allows us to interpret the housing market and make predictions about the selling price of new homes, based on the linear function:

Y=hΘ(x)=X×ΘT

Where hΘ(x) is the linear hypothesis.

**Creation of a synthetic dataset**

First, we will create an example of a house with known data, with the values of its features and the sale price:

In [ ]:

x\_ex1 **=** np **.** asarray([100, 4, 2, 2, 1, 0, 30, 10])

y\_ex1 **=** np **.** asarray([80000])

print('Sale price of the house:', y\_ex1[0])

print('Useful surface area:', x\_ex1[0])

print('Location:', x\_ex1[1])

print('Dwelling type:', x\_ex1[2])

print('Nº of rooms:', x\_ex1[3])

print('Nº of bathrooms:', x\_ex1[4])

print('Garage (yes/no):', x\_ex1[5])

print('Surface area of common areas:', x\_ex1[6])

print('Age', x\_ex1[7])

In this way, we can create new examples with the values we want as needed.

Modify the values of the previous dwelling to manually generate other dwellings.

In the same way that we have manually created a housing example, we will automatically create multiple random examples:

_Note:_ For the sake of simplicity when generating random numbers with code, we will use _float_ instead of _int_ in the same logical ranges for the features of the dwellings.

In [ ]:

m **=** 100 _#nº of housing examples_

n **=** x\_ex1 **.** shape[0] _#nº of features of each dwelling_

X **=** np **.** random **.** rand(m, n)

print('First 10 examples of X:')

print(X[:10, :])

print('Size of the matrix of X examples:')

print(X **.** shape)

_How can we create the vector_ Y _of sales prices from our synthetic dataset, so that it follows the linear relationship we want to approximate?_

To do this, we must start from a _Theta\_true_ as in previous exercises.

Sometimes, the problem is to get a _Y_ in the range we want by modifying each value of _Theta\_true_, which can be quite tedious.

In other cases, an alternative would be to manually create _X_ and _Y_, and then calculate the _Theta\_true_ corresponding to those matrices.

In this case, we will manually create _Theta\_true_ in order to be able to interpret its values:

In [ ]:

x **=** X[0, :]

print('Example of dwelling with random features:')

print(x)

Theta\_true **=** np **.** asarray([1000., **-** 500, 10000, 5000, 2500, 6000, 50, **-** 1500])

print('\nEx. of manually created feature weights:')

print(Theta\_true)

print(Theta\_true **.** shape)

print('\nThe selling price of this property is given by its features:')

print('for each m2 of usable surface area:', Theta\_true[0])

print('For each km to the town center:', Theta\_true[1])

print('According to dwelling type:', Theta\_true[2])

print('According to the number of rooms:', Theta\_true[3])

print('According to the number of bathrooms:', Theta\_true[4])

print('Depending on whether it has a garage:', Theta\_true[5])

print('For each m2 of common areas:', Theta\_true[6])

print('For each year in age:', Theta\_true[7])

Each of these weights will multiply its corresponding feature, adding to or subtracting from the total price of the dwelling.

However, our ideal synthetic dataset is missing a very important term, thebias or intercept term: The bias is the _b_ term of any line y=m∗x×b, representing the sum of all the constants that affect our model or the base price, before being modified by the other features.

Thisbias is very important because a model must not only be able to approximate weights or coefficients that multiply the given features, but also constant values that do not depend on the specific features of each example.

In other words: _price = coefficients \* features + "bias or base price"_.

E.g., in the case of dwellings, it would be the starting price of all the dwellings, irrespective of their features, if any, which would add to or subtract from it. In the case of a studio without independent rooms, shared bathroom, no garage, etc., etc., where all its features were 0, it allows us to determine its selling price, _which would certainly not be 0 €._

We add abias or starting price to _Theta\_true_.

In [ ]:

_# CAUTION: Each time we execute this cell we are adding a column_

_# of 1's to Theta\_true and X, so we only need to run it once_

Theta\_true = np.insert(Theta\_true, 0, 25000) _#__25000 € starting price = theta[0]_

X **=** np **.** insert(X, 0, np **.** ones(m), axis **=** 1)

print('Theta true and first 10 examples (rows) of X:')

print(Theta\_true)

print(X[:10, :])

print('Sizes of X and Theta true:')

print(X **.** shape)

print(Theta\_true **.** shape)

With this _Theta\_true_, we can establish the vector _Y_ of sale prices for our examples:

In [ ]:

_# TODO: Modify the following code to add a random error term to Y_

error **=** 0.15

Y **=** np **.** matmul(X, Theta\_true)

print('Sale price:')

print(Y)

print(Y **.** shape)

_Note:_ Since in the end no _int_ values were used, the sale prices are alsofloat values.

**Training the model**

Once our training data dataset is ready, we will train the linear regression model.

To do this, copy the corresponding cells from the last exercises to train the model with this data and evaluate its behaviour:

In [ ]:

_# TODO: Copy the corresponding cells to train a linear regression model and evaluate its training_

**Predictions**

Therefore, if we manually create a new house example with random features, we can make a prediction about its selling price:

In [ ]:

_# TODO: Create a new dwelling with random features and calculate its predicted Y_

_# Remember to add a bias term of 1 to X._

x\_pred **=** [**...**]

y\_pred = np.matmul(x\_pred, theta) _# Use the theta trained for your model in the previous step_

print('Random housing example:')

print(x\_pred)

print('Predicted price for this house:')

print(y\_pred)

What about our original synthetic dataset, how much would it sell for, and what would be the residuals of our model on them?

In [ ]:

_# TODO: Calculate and plot the model's residuals_

Y\_pred **=** [**...**]

residuals **=** [**...**]

plt **.** figure()

_# Give a name to the graph and label the axes_

plt **.** show()
