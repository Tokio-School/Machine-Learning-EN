# **K-means: Clustering on the Iris dataset and comparison with classification**

M3U2 - Exercise 2

**What are we going to do?**

- We will download the Iris dataset
- We will train a K-means clustering model using Scikit-learn
- We will evaluate the results of the model graphically
- We will train a classification model on the same dataset using SVM
- We will evaluate the results of the classification
- We will compare both models and their results

Remember to follow the instructions for the submission of assignments indicated in [Submission Instructions](https://github.com/Tokio-School/Machine-Learning/blob/main/Instrucciones%20entregas.md).

**Instructions**

In this exercise, we will compare both algorithms on the Iris dataset, which we had previously used in classification. In this way we will be able to compare both types of learning; supervised classification learning and unsupervised clustering learning.

We may often confuse the two types of models or think that we can use them for similar use cases. However, we must always remember the fundamental difference between them: that classification is supervised learning (where we have previously annotated and known results) and clustering is unsupervised (we do not have results).

In [ ]:

_# TODO: Import all the necessary modules into this cell_

**Download the Iris dataset**

Download the Iris dataset in _X_ and _Y_ format for use in this exercise.

You can use this link: [sklearn.datasets.load\_iris](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html).

In [ ]:

_# TODO: Download the Iris dataset_

_Do you have the courage to represent the first 3 features in 3D? Sepal length, sepal width, and petal length._

_Is there a clear differentiation between the classes? Do you see multiple data clusters?_

You can use these exercises as a reference:

- [The Iris Dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html)
- [3D scatterplot](https://matplotlib.org/3.1.1/gallery/mplot3d/scatter3d.html)
- [K-means Clustering](https://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_iris.html)

_NOTE:_ We will use these same 3 features and the same 3D rendering technique when rendering results throughout the exercise.

In [ ]:

_# TODO: Represent 3 features from the 4-dimensional dataset in 3D_

To better analyse the dataset, try to represent another set of 3 different features, also including the petal width.

Among these combinations of features, represented on different graphs, _can you find a clear difference between classes? Several clusters of data grouped together? More descriptor variables?_

In view of this data, which may also be difficult to appreciate for our models, we will check their results.

**Preprocess the data**

We will preprocess the data in the usual way:

- Randomly reorder the examples.
- Normalise them only if necessary.
- Divide them into training and test subsets.

We will use the training subset exclusively to train our models and the test subset to evaluate them, with cross-validation by K-means. This will ensure that both types of models are trained and evaluated on equal terms.

In [ ]:

_# TODO: Randomly reorder the data, normalise it if necessary, and split it into training and test subsets._

**Train a K-means clustering model using Scikit-learn**

We will train a K-means clustering model using the Scikit-learn method. [sklearn.cluster.KMeans](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html).

This method accepts an _n\_clusters_ number of clusters and performs several initialisations with different original centroids, so we will not have to perform cross-validation on it.

Train a K-means model for 3 clusters on the training subset:

In [ ]:

_# TODO: Train a K-means model for 3 clusters on the training subset:_

**Evaluate your results graphically**

On this occasion, to evaluate the K-means model, we will use the final test subset since we have it.

Evaluate this model with its model.score() and silhouette coefficient:

_NOTE:_ Although implemented for consistency, model.score() does not use _Y_.

In [ ]:

_# TODO: Evaluate this model without using Y on the test subset_

Plot the results:

In [ ]:

_# TODO: Plot the centroids and the examples assigned to each cluster in 3D_

**Train an SVM classification model on the Iris dataset**

Now train an SVM classification model using [sklearn.svm.SVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html) on the Iris dataset.

Evaluate various values of the _C_ hyperparameter and available kernels using K-fold with [sklearn.model\_selection.GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html).

For _gamma_ you can use the default value or auto.

You can use these exercises as a reference: [Plot different SVM classifiers in the Iris Dataset](https://scikit-learn.org/stable/auto_examples/svm/plot_iris_svc.html).

In [ ]:

_# TODO: Train an SVC model, optimising its hyperparameters and kernel using CV with K-fold_

**Evaluate your results graphically**

Finally, as we did in the exercises of the previous session, evaluate the model on the test subset and plot its predictions graphically:

In [ ]:

_# TODO: Evaluate the model on the test subset_

In [ ]:

_# TODO: Plot its predictions graphically in 3D_

**Compare both models and their results**

Although the two tasks, clustering and classification, are inherently different, we can compare these results to highlight their differences.

To do this, you can compare the assignment of examples to each cluster in grouping and to each class in classification.

_Note:_ Please consider the inherent complexity of the examples and features of this Iris dataset.

- _Is there a big difference between the number of examples that would be assigned to a cluster in clustering and to a different class in classification?_
- _Which model is more accurate? Is a clustering model able to deliver similar results to a classification model, when we can evaluate the actual class?_
- _How is the partitioning of the n-dimensional space in both cases? Is the space assigned to classes and the space assigned to clusters similar in the graphical representations?_
