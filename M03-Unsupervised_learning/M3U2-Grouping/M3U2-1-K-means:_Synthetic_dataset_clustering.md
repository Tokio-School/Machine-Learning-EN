# **K-means: Synthetic dataset clustering**

M3U2 - Exercise 1

**What are we going to do?**

- Create a synthetic clustering dataset
- Preprocess the data
- We will implement the K-means clustering algorithm and test the implementation
- We will train a K-means model with multiple initialisations
- We will evaluate the model and represent its results graphically
- We will choose an optimal number of clusters using the elbow rule

Remember to follow the instructions for the submission of assignments indicated in [Submission Instructions](https://github.com/Tokio-School/Machine-Learning/blob/main/Instrucciones%20entregas.md).

**Instructions**

In this exercise we will implement a K-means clustering model training algorithm.

To do this, we will create a synthetic clustering dataset, develop our K-means implementation and test it.

As we know, clustering models are very sensitive to initialisation conditions, so we are going to train several models and check graphically whether their results are noticeably different.

We will evaluate them graphically, plotting the final result.

In [ ]:

_# TODO: Import all the necessary modules into this cell_

**Create a synthetic dataset**

We are going to create a synthetic classification dataset. This dataset will have a set number of clusters with a number of examples and an associated standard deviation.

To make it easier, you can use the [sklearn.datasets.make\_blobs function](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_blobs.html).

In [ ]:

_# TODO: Create a synthetic clustering dataset_

n\_samples = 1000

n\_features = 2

centers **=** 5

cluster\_std **=** [1., 1.5, 0.5, 2., 3.]

return\_centers **=**** True**

random\_state **=** 42

X, y **=** [**...**]

_# Display the first X rows and columns and their dimensions_

print('First 10 rows and 5 columns of X and Y and their dimensions:')

print()

print()

print()

print()

**Preprocess the data**

Preprocess the data with the usual steps, if necessary:

- Randomly reorder the data
- Normalise the data

As clustering is an unsupervised learning model, we will not evaluate it by the conventional method, i.e., comparing its results with previously known results.

In [ ]:

_# TODO: Randomly reorder the data_

In [ ]:

_# TODO: Normalise the data, if necessary_

**Implement the K-means algorithm**

Now implement the K-means clustering algorithm.

Remember the steps of the algorithm:

1. Define the Nº of clusters to be considered
2. Initialise the centroids of each cluster, e.g., by choosing the first _k_ examples of the dataset.
3. Assign each example in the dataset to its nearest centroid.
4. Calculate the midpoint of each cluster in the n-dimensional space of the dataset.
5. Update the centroid corresponding to that point.
6. Reassign each example to its new nearest centroid.
7. Continue iterating until the training converges: the centroids do not vary in position, or vary less than the given tolerance, or we reach the maximum number of iterations.

The output of the model will be the training data with its nearest assigned centroid, and the position of the centroids:

In [ ]:

_# TODO: Implement an auxiliary function to calculate the distance between the examples and a given point_

**def** dist\_examples(x, centroid):

""" Calculates the Euclidean distance between point x and the centroid in n-dimensional space

Arguments:

x -- 1D ndarray with the features of the example

centroid -- 1D ndarray with the location of the centroid

Return:

dist -- Euclidean distance between x and the centroid in n-dimensional space

"""

**return** dist

In [ ]:

_# TODO: Implement the K-means clustering algorithm_

n\_clusters **=** 5

n\_iter **=** 100

tol **=** 1e-3

_# Initialise the centroids as a 2D ndarray with the n-dimensional position of the first n\_clusters examples, size (n\_clusters x n)_

centroids **=** [**...**]

_# Iterate over the maximum nº of iterations_

**for** i **in** range(n\_iter):

_# Assign each example to its nearest centroid using dist\_examples()_

**for** x **in** n\_samples:

cluster\_assigned\_examples = [...] _# size m, values [0, n\_clusters - 1], according to the centroid closest to each example_

_# Calculate the n-dimensional midpoint for each cluster with its assigned examples_

**for** c **in** n\_clusters:

**for** n **in** n\_features:

_Tips: You can use Numpy functions to calculate the average of an array or a slice of an array_

centroid[**...**] **=** [**...**]

_# Update the centroid of each cluster to that midpoint_

centroids[**...**] **=** centroid

_# Check if the model converges: all the centroids move less than the tolerance_ tol

**if** [**...**]:

print('Model converges at iteration no.:', i)

**break**

**else** :

print('Max. number of iterations reached')

print('Final centroid locations:')

print(centroids)

print('Centroids assigned to each example (0-{}):'.format(n\_clusters - 1))

print(cluster\_assigned\_examples)

_Note:_ Remember that the code template cells are always just suggested guidelines for implementing your code. If you prefer to modify them to develop your code with a different structure, you may do so at any time. The only important thing is that the final calculation is correct and returns the final results to be reviewed.

**Evaluate and plot the results**

We will plot a 2D graph with the results of our training: the centroid of each cluster and the examples assigned to it. Similarly, we will use appropriate evaluation metrics for clustering (different from those used for classification).

For this purpose, you can use this example as a reference: [A demo of K-Means clustering on the handwritten digits data](https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_digits.html).

By creating the original dataset, we have retrieved the centroid of each cluster used to create it, and the _ground truth_ of this dataset, which we can use in this evaluation.

In our case, our _n_ (number of dimensions) or _n\_features_ is also 2, so we do not have to reduce the dimensionality (we will introduce this concept in a later session).

In [ ]:

_# TODO: Evaluate your model with the metrics of homogeneity, completeness, V-measure, adjusted Rand index, adjusted mutual information index, and the silhouette coefficient_

_Note:_ Take the opportunity to dive into the literature and learn more about these coefficients, used to evaluate clustering, which are different from those used in classification: [Clustering performance evaluation](https://scikit-learn.org/stable/modules/clustering.html#clustering-evaluation).

In [ ]:

_# TODO: Represent your trained model: the centroids of each cluster and the examples assigned to each one_

**Train a K-means model with multiple initialisations**

As mentioned above, K-means clustering is an algorithm that is quite sensitive to the initialisation used, as the final result may vary. To check this graphically, retrain the model again, choosing different initial centroids, for which you can randomly reorder the data.

To do this, copy the code cells below that train the model, evaluate it and plot its results graphically. This way you can compare the results of both cases at the same time.

_Note:_ For the graphical representation, change the number of the figure to _plt.figure(1)_.

In [ ]:

_# TODO: Implement the K-means clustering algorithm_

In [ ]:

_# TODO: Evaluate your model with the metrics of homogeneity, completeness, V-measure, adjusted Rand index, adjusted mutual information index, and the silhouette coefficient_

In [ ]:

_# TODO: Represent your trained model: the centroids of each cluster and the examples assigned to each one_

_How do the results of your model vary with another random initialisation, its evaluation metrics, and the graph of results?_

You can also recreate the original dataset, even changing the size or standard deviation of each cluster, and see if it affects the results and the variance between clusters.

**Multiple models in parallel**

We will now train multiple models in parallel with different initialisations and compare their results.

To do this, copy and modify the corresponding cells again, and train 10 different models, on the same data, with 10 different random initialisations of the centroids.

Finally, graphically compare the adjusted Rand index for all the models:

In [ ]:

_# TODO: Train 10 models on the same data with 10 random centroid initialisations_

In [ ]:

_# TODO: Graphically represent the comparison of their adjusted Rand indices on a line and dot plot._

**Choosing the optimal number of clusters**

Having created a synthetic dataset, we have chosen the "correct" number of clusters for it. However, in a real dataset we will not know this number, and on many occasions, there will not be a correct number of clusters, since detecting the separation between one cluster or another, if they are very close, can be a non-trivial and subjective task.

By mathematical logic, the lower the number of clusters, the greater the average distance between each example and its assigned cluster, and the higher the number of clusters, the smaller the distance. In a reductio ad absurdum, when we use a number of clusters equal to the number of examples, each centroid will ideally correspond to the position of each example, and the average distance to the nearest cluster will be 0.

Therefore, to choose the optimal number of clusters when we do not have any external considerations or constraints, we can use the so-called "elbow rule".

Let's apply this rule for our dataset:

1. Train a model for each number of clusters to be considered in a range, e.g. [1, 10], more in a case where n\_clusters = n\_samples.
2. For each number of clusters, we train several models with multiple random initializations, and we choose the one with the best [silhouette coefficient](https://scikit-learn.org/stable/modules/clustering.html#silhouette-coefficient)
3. We plot the best evaluation metric of the best model vs. the number of clusters considered
4. We choose as the "optimal" number of clusters the one where there is a "bend" in the graph, where the trend or slope changes most abruptly.

In a real dataset we will not have its _ground truth_, the correct centroids, so as an evaluation we will use the silhouette coefficient metric.

Implement the elbow rule to choose an optimal number of clusters for this dataset:

In [ ]:

_# TODO: Implement the elbow rule to choose an optimal number of clusters_

n\_clusters **=** [**...**] _# Array [1, 2, 3, ..., 10, n\_samples]_

n\_iter **=** 100

tol **=** 1e-3

_# Iterate over the nº of clusters to be considered_

**for** n\_c **in** n\_clusters:

_# Train various models with random initialisations_

**for** \_ **in** range(5):

_# Use your modified code from previous cells to train the models_

[**...**]

_# Evaluate each model using the silhouette coefficient and keep the best one_

_# Pseudo-code_

**if** evaluation **\>** best\_evaluation:

best\_model **=** model

_# As a final result we seek:_

print('Centroids of each model, according to the number of clusters:')

print()

print('Silhouette coefficient of each model, according to the number of clusters:')

print()

In [ ]:

_# TODO: Plot the elbow rule on a line and dot plot: the silhouette coefficient of the best model vs. the number of clusters considered._

plt **.** figure()

plt **.** plot([**...**])

Choose an optimal number of clusters for your final result and indicate it in this cell:

- Optimal number of clusters by the elbow rule: X

_What if we change the original number of clusters in the dataset, does the optimal number still show up as clearly on the graph?_

Modify the original dataset and rerun the elbow rule resolution for comparison. Use several different cluster numbers.

Finally, plot the results of the selected model again, with the optimal number of clusters:

In [ ]:

_# TODO: Represent your trained model: the centroids of each cluster and the examples assigned to each one_
