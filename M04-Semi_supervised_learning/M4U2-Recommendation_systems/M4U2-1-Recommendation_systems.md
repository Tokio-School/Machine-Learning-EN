# **Recommender systems: Collaborative filters**

M4U2 - Exercise 1

**What are we going to do?**

- We will investigate the collaborative filtering approach
- We will create a dataset to be solved by recommender systems
- We will implement the cost and gradient descent functions
- We will train a recommendation model using collaborative filters
- We will make predictions of recommendations
- We will retrain the model by incorporating new valuations
- We will recommend similar examples to others

In [ ]:

_# TODO: Use this cell to import all the necessary libraries_

**import** time

**import** numpy **as** np

**import** matplotlib.pyplot **as** plt

**from** scipy.spatial **import** distance

np **.** random **.** seed(42)

# **Create a synthetic dataset**

A common example is film recommendations on a video streaming portal. In this case, a dataset would have these features e.g.:

- _m:_ Nº of films.
- _n_: Number of features of each film and coefficients of each user for them.
- nu: Nº of portal users.
- nru and nr: Percentage of ratings for each film and total number of ratings, known in advance.
- _X_: 2D matrix of features for each film, size (no of films, no of features).
- Θ: 2D matrix of coefficients of each user for each film, size (nº of features, nº of users).
- _Y_: 2D Matrix of ratings of each user for each film, size (nº of films, nº of users).

We are going to create a synthetic dataset as usual, but this time focused on recommender systems, with some differences compared to linear regression:

- The predictor or independent features _X_ (size (_m, n_ + 1)), which represents the features of each example, **are not known in advance**.
- The vector Θ (_Theta_) is 2D (size (_n_ + 1, nu)), since it now represents the coefficients of the features for each user. Again, **it is not known in advance**.
- The vector _Y_ is 2D (size (_m_, nu)), as it now represents each user's rating for each example.
- Vector _Y_ will contain both the "real" ratings given by each user for each film they have rated, and, at the end of the training, their predicted ratings for recommending one film or another.
- _R_ will be a "mask" matrix over_Y_, used to indicate which ratings of _Y_ are real and issued by a user, and therefore only those used to train the model.

To have at hand, we leave you this quick reference table with the size of each matrix:

- X (m, n+1)
- Θ (n+1, nu)
- Y (m, nu)

In order not to complicate the implementation further, we will not preprocess the data in this case.

Follow the instructions to generate a dataset with the necessary features to be solved by a collaborative filter:

In [ ]:

_# TODO: Create a dataset with the necessary features for a recommender system_

_# Remember that you can go back to this cell and modify the features of the dataset at any time_

m **=** 1000 _# Nº of examples_

n = 4 _# Nº of features for each example/user_

n\_u **=** 100 _# Nº of users_

n\_rk **=** 0.25 _# Percentage of valuations known in advance_

_# Create an X with random values and size (m, n)_

_# Insert a column of 1's in the first column_

X\_true **=** [**...**]

_# Create a Theta\_true with random values and size (n + 1, n\_u)_

Theta\_true **=** [**...**]

_# Create a Y\_true of size (m, n\_u) by multiplying X\_true and transposed Theta\_true_

Y\_true **=** [**...**]

_# Create an R matrix of 0's with size (m, n\_u)_

r **=** [**...**]

count\_r **=** round(n\_kr **\*** r **.** size) _# nº of known valuations or 1's in R_

**while** count\_r:

_# Generate a random int between [0, m] as an index of R_

i **=** [**...**]

_Generate a random int between [0, n\_u] as an index of R_

j **=** [**...**]

_# Change that element from R to 1 if it has not been changed before and subtract 1 from the number of known ratings_

**if**** not** r[i, j]:

r[i, j] **=** 1.

count\_r **-=** 1

_# Count values of R other than 0._

n\_r **=** [**...**]

_# Generate a Y with only the known valuations using R_

y **=** [**...**]

print('Size of X(m, n+1), Theta(n+1, n\_u) and Y(m, n\_u) true:')

print(X\_true **.** shape, Theta\_true **.** shape, Y\_true **.** shape)

print('Size of y and R known in advance:')

print(y **.** shape, r **.** shape)

print('Nº of elements of R or known valuations:', n\_r)

# **Cost and gradient descent functions**

We will implement the regularized cost function and gradient descent to train the ML model.

Conceptually, we will follow different steps from linear regression:

Whereas in linear regression _Y_ and _X_ were known, and we could iteratively optimize Θ to reduce the cost, here _X_ is not known in advance, as it is usually impossible in practice to know or have all the features of all the examples or films written down in advance.

Also, while we do have some user ratings for some films, we typically have a fairly low percentage of ratings for each example, so _Y_ is not completely known beforehand and most of its values will be initially empty.

Therefore, our goal will be, not to solve Θ, but rather _Y_, to fill it by obtaining all the predicted ratings of each user for each example.

Therefore, the training algorithm will be:

1. We collect the examples in matrices _X_, _Θ,_ and _Y_.
2. We mark the known valuations in thesparse _R_ matrix.
3. Given _X_ and _Y_, we can obtain Θ.
4. Given Θ and _Y_, we can obtain _X._
5. We iteratively estimate _X_ and Θ at each iteration until the training converges to a minimum cost.
6. When more valuations are available, we retrain the model by adding them to _Y_ and marking them in _R_.

In the next cell, follow the instructions to implement the regularized cost function and gradient descent for a collaborative filter, following the formulas below:

minθ0,...,θnu,x0,...,xnmJ(x0,...,xnm,θ0,...,θnu)=minθ0,...,θnu,x0,...,xnm[12∑(i,k):r(i,k)=1(xi×θkT−yki)2+λ2∑i=0n∑k=0nu(xki)2+λ2∑j=0n∑k=0nu(θkj)2]xki:=xki−α(∑j:r(i,k)=1(xi×θkT−yki)θkj+λxki);θkj:=θkj−α(∑i:r(i,k)=1(xi×θjT−yki)xki+λθkj); j=0→λ=0

In [ ]:

_# TODO: Implement the cost function for collaborative filters_

**def** cost\_function\_collaborative\_filtering\_regularized(x, theta, y, r, lambda\_ **=** 0.):

_# TIPS: Plot the operations step by step on paper, noting the dimensions of the original vectors and those of the result of each intermediate operation._

_# Use ndarray.reshape() if you need to, especially with 1D vectors (e.g. (6,)) that can give you unexpected results in Numpy_

_# Use m, n, and n\_u in ndarray.reshape(), not "hard-coded" values like 6, 20, etc._

_# Use np.matmul() to multiply matrices_

_# To train on only known values, multiply R by the result of the subtraction of the hypothesis e and the mask matrix_

_# In choosing the slices or vectors of X, Theta and Y correctly, there is no major difference with linear regression_

j **=** [**...**]

_# Calculate the regularisation factor for X_

x\_reg **=** [**...**]

_# Calculate the regularisation factor for Theta_

_# Remember not to regularise the first column_

theta\_reg **=** [**...**]

j **=** [**...**]

**return** j

**Check the implementation of the cost function**

Check your implementation of the cost function in the following scenarios:

1. If theta = Theta\_true, j = 0
2. If theta = Theta\_true and lambda\_ != 0, j != 0
3. The further lambda\_ moves away from 0  or theta, the more j increases
4. If all the elements of r are 0, then no elements are considered for training, therefore j = 0.

In [ ]:

_# TODO: Check the implementation of the cost function_

Record your results in this cell:

1. Experiment 1
2. Experiment 2
3. Experiment 3
4. Experiment 4

In [ ]:

_# TODO: Implement gradient descent training for collaborative filters_

**def** gradient\_descent\_collaborative\_filtering\_regularized(x, theta, y, r, lambda\_ **=** 0., alpha **=** 1e-3, n\_iter **=** 1e3, e **=** 1e-3):

_# To train on only known values, multiply R by the result of the subtraction of the hypothesis e and the mask matrix_

n\_iter **=** int(n\_iter) _# Convert n\_iter to int so it can be used in range()_

_# Initialise j\_hist with the vlaues from the history of the cost function_

j\_hist **=** []

_# Add as first value the cost of the cost function for the initial values_

j\_hist **.** append(cost\_function\_collaborative\_filtering\_regularized([**...**]))

**for** iter\_ **in** range(n\_iter):

_# Initialise some empty theta and x to fill in the gradient with ndarrays of the same size as the original ones_

_# and empty vector values (more optimized), zeros or random, so as not to modify theta, which must be kept constant during the iteration iter\__

theta\_grad **=** [**...**]

x\_grad **=** [**...**]

**for** k **in** range(n\_u):

_# Calculate the gradient to update theta at this iteration_

_# Use theta and not theta\_grad in the intermediate operations, since we want to modify theta\_grad and not the original theta_

theta\_grad[:, k] **=** [**...**]

_# For all theta\_grad, except the first column, add the regularization term_

theta\_grad[1:, k] **+=** [**...**]

**for** i **in** range(m):

_# Calculate the gradient to update X at this iteration_

_# Follow similar steps to the theta gradient to implement the corresponding function_

_# Add the regularisation term_

x\_grad[i, :] **=** [**...**]

_# Update X and Theta with the gradients_

x **-=** alpha **\*** x\_grad

theta **-=** alpha **\*** theta\_grad

_# If you need to, check how X and Theta are being updated_

_#print('\nUpdated values of X and Theta ')_

_#print(x)_

_#print(x.shape)_

_#print(theta)_

_#print(theta.shape)_

_# Calculate the cost at this iteration and add it to the cost history_

j\_cost **=** regularized\_cost\_function\_collaborative\_filtering([**...**])

j\_hist **.** append(j\_cost)

_# If it is not the first iteration and the absolute difference between the cost and that of the previous iteration is_

_less than e, declare convergence_

**if** [**...**]:

print('Converge at iteration nº, iter\_)

**break**

**else** :

print('Max. nº of iterations reached' **.** format(n\_iter))

**return** j\_hist, x, theta

# **Training the model**

Once the corresponding functions have been implemented, we will train the model.

To do this, complete the following code cell with steps equivalent to other models from previous exercises.

In [ ]:

_# TODO: Train a collaborative filter recommendation system model_

_# Generate an initial X and Theta with random values and the same size as X\_true and Theta\_true_

x\_init **=** [**...**]

theta\_init **=** [**...**]

alpha **=** 1e-2

lambda\_ **=** 0.

e **=** 1e-3

n\_iter **=** 1e4

print('Hyperparameters used:')

print('Alpha:', alpha, 'Lambda:', lambda\_, 'Error:', e, 'Nº iter', n\_iter)

t0 **=** time **.** time()

j\_hist, x, theta **=** regularized\_gradient\_descent\_collaborative\_filtering([**...**])

print(Training duration:', time **.** time() **-** t0)

print('\nLast 10 values of the cost function')

print(j\_hist[**-** 10:])

print('\nError final:')

print(j\_hist[**-** 1])

As we have done on previous occasions, plot the evolution of the cost function to check that the training of the model has been done correctly:

In [ ]:

_# TODO: Plot the model training cost function vs. the number of iterations_

plt **.** figure()

plt **.** plot([**...**])

_# Add a title, label both axes of the graph, and a grid_

[**...**]

plt **.** show()

**Check the implementation of the gradient descent**

Check your model training implementation in the following scenarios:

1. If theta = Theta\_true, the model converges in 1 or 2 iterations with a final cost j = 0
2. The further theta moves away from Theta\_true, the higher the intermediate cost and the more iterations until the model converges
3. The more elements r has, the less time it takes to converge and the lower the final cost of the model.

In [ ]:

_# TODO: Check the implementation of gradient descent_

Record your results in this cell:

1. Experiment 1
2. Experiment 2
3. Experiment 3

# **Making predictions of recommendations**

Once the model has been trained, we can solve the recommendation matrix _Y_, which contains both the ratings issued by the users and a prediction of each user's rating for each example.

Remember that we used the _R_ matrix to mark actual valuations with a 1, while those that have been predicted and were not known beforehand are marked with a 0.

To make a prediction and recommend examples to users (e.g., films), follow the instructions to complete the following code cell:

In [ ]:

_# TODO: Make predictions of examples for users_

_# Display the ratings of the Y matrix_

print('Pre-known values (first 10 rows and columns):')

print(y[:10, :10] \* r[:10, :10]) _# Limit the number of rows and columns of Y to display it on screen_

_# Show more or fewer rows and columns if necessary to validate your model_

_# In the result, a value of "0." indicates a "0" at that position in R, or that this initial valuation is not known_

_# Calculate the predictions obtained by the model from X and Theta_

y\_pred **=** [**...**]

print('\nPredicted ratings (first 10 rows and columns):')

print(y\_pred[:10, :10])

_# Calculate the residuals for the predictions_

_# Remember that the residuals are the difference in absolute value between the previously known true value and the model predictions_

_# Remember to calculate them only when the initial valuation is known, multiplying the residuals by R_

y\_residual **=** [**...**]

print('\nModel residuals (first 10 rows and columns):')

print(y\_residual[:10, :10])

_# Display the initial predictions and ratings of a given user._

jj = 0 _# Choose a user index between 0 and n\_u_

print('\nActual and predicted ratings for user no. {}:'.format(jj + 1))

print(y\_pred[:, jj])

_# Sort the indexes of the examples we would recommend to each user according to their ratings, in descending order_

_# Remember to remove the ratings initially issued by the user from the list, or films already viewed, those whose R[i, k] = 0_

_# You can sort a ndarray with numpy.sort()_

print('\nPredicted ratings for user no. {}:'.format(jj + 1))

print([**...**])

_# You can get the indices that would sort a ndarray with numpy.argsort()_

y\_pred\_ord **=** [**...**]

print('\nIndices of the examples to recommend to the user {}, according to their predicted ratings:'.format(jj + 1))

print(y\_pred\_ord)

# **Retraining by incorporating new valuations**

To retrain the model by incorporating new user ratings, simply modify the initial _Y_ with the new ratings and mark the position in the _R_ matrix with a 1.

Follow the instructions in the next cell to add new ratings:

In [ ]:

_# TODO: Add 2 new user ratings to 2 examples of your choice_

_# Choose a user index and an example index_

i\_1 **=** 2

k\_1 **=** 2

i\_2 **=** 3

k\_3 **=** 3

_# Choose a rating. They usually take values between [0, 2)_

y[**...**] **=** 1.

y[**...**] **=** 1.

_# Mark them as new valuations in R_

r[**...**] **=** 1.

r[**...**] **=** 1.

Now re-train the model by re-executing the training cell and the subsequent cells, up to the previous cell.

Check that these positions now show the new valuation and not a prediction of the model.

# **Find similar examples and users**

To find the similarity between 2 elements, we can compute the Euclidean distance between them.

The Euclidean distance in this n-dimensional space will represent the cumulative difference between the coefficients of these elements, just as a distance in a 2D or 3D plane is the cumulative difference between the coordinates of those points.

Find similar examples and users by following the instructions in the next cell:

In [ ]:

_# TODO: Find examples and users similar to each other_

_# Calculate the similarity between the first 4 (X) examples_

dist\_ex **=** distance **.** cdist([**...**])

print('Similarity between the first 4 examples:')

print(dist\_ex)

_# Calculate the similarity between the first 4 users (Theta)_

dist\_us **=** distance **.** cdist([**...**])

print('Similarity between the first 4 users:')

print(dist\_us)

_# Calculate the example most similar to the first_

index\_ex\_similar **=** [**...**]

ex\_similar **=** [**...**]

print('Coefficients of example no. {} for the first 5 users:'.format(0 + 1))

print(x[0, :5])

print('The example most similar to example no. {} is example no. {}'.format(0 + 1, index\_ex\_similar))

print('Coefficients of example no. {} for the first 5 users:'.format(index\_ex\_similar))

print(ex\_similar[:5])

_# Calculate the user most similar to the first_

index\_us\_similar **=** [**...**]

us\_similar **=** [**...**]

print(Coefficients of user nº {} for the first 5 examples:'.format(0 + 1))

print(theta[0, :5])

print('The user most similar to user no. {} is user no. {}'.format(0 + 1, index\_us\_similar))

print('Coefficients of user nº {} for the first 5 examples:'.format(index\_us\_similar))

print(us\_similar[:5])

**Bonus: Check what happens if we do not have sufficient initial assessments**

_What happens if we don't have a minimum number of ratings initially, what if there is an example that has no ratings from any user, or a user who has not rated any examples?_

_Do you think that, in that case, we could train the model and get results for those examples and users?_

To check this, you can e.g., decrease the percentage of initial assessments to a value that is too low, e.g., 10%, and check what happens to the evolution of the training cost function.
