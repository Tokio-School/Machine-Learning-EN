# **Comparison of clustering algorithms with Scikit-learn**

M3U2 - Exercise 3

**What are we going to do?**

- We will compare the different clustering algorithms available in Scikit-learn
- We will check the initial assumptions for the case of K-Means

Remember to follow the instructions for the submission of assignments indicated in [Submission Instructions](https://github.com/Tokio-School/Machine-Learning/blob/main/Instrucciones%20entregas.md).

**Instructions**

In this exercise you will not develop new code, but simply download and run 2 notebooks from the Scikit-learn documentation:

- [Comparing different clustering algorithms on toy datasets](https://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_comparison.html)
- [Demonstration of k-means assumptions](https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_assumptions.html)

There are many conclusions we could add, e.g.:

- No single clustering algorithm works well in all cases, with all possible data distributions.
- Prior data analysis is even more critical when training clustering algorithms.
- Pre-visualisation will help a lot, although it is quite difficult without dimensionality reduction.
- The problem with clustering is that it is often inherently difficult to solve.

Copy the code cells from the examples in the documentation, analyse the results, modify some parameters to experiment and test your intuition and answer the questions at the end of the exercise.

**Comparing different clustering algorithms on sample datasets**

The clustering problem is an inherently difficult problem to solve accurately, since there are no known prior results that we can use to monitor our learning, and choosing the right clusters can be quite difficult, in addition to always being subjective.

There are many different clustering algorithms we can use, in addition to K-means and its variants, and each of them will perform better or worse depending on the data distribution of the dataset.

Copy the code of the exercise **Comparing different clustering algorithms on toy datasets** and see for yourself.

In [ ]:

_# TODO: Execute the code of the corresponding exercise_

**Demonstration of the assumptions of the K-means algorithm**

As a general rule, the K-means algorithm and its variations will perform poorly on datasets with the following distributions of examples:

- An incorrect number of blobs or clusters used to train the model.
- Datasets with non-spherical shapes close together.
- Variance between nearby clusters that are very different from each other.

In contrast, it works quite well on separate clusters with similar variance, even if the number of examples is very different.

Copy the code of the exercise **Demonstration of k-means assumptions** and see for yourself.

In [ ]:

_# TODO: Execute the code of the corresponding exercise_

Modify the parameters of the dataset used to see if the assumptions hold true.

**Questions to answer**

Please answer the following questions by adding a new cell after this Markdown cell with your answers following the same list. We have divided the questions into 2 blocks, each relating to one of the Scikit-learn documentation notebooks, so when we mention datasets and algorithms we are referring to those in the Scikit-learn notebook.

**Comparing different clustering algorithms on sample datasets**

1. _QUESTION_: Can you describe in your own words the 6 different datasets used?
  1. Note: This may seem like a silly question, but although the differences are obvious, having to pause and describe the differences helps you to really look at them, and documenting it will help you to see if it fits with some features of another real-life dataset.
  2. Dataset 1:
  3. Dataset 2:
  4. Dataset 3:
  5. Dataset 4:
  6. Dataset 5:
  7. Dataset 6:
2. _QUESTION:_ Compare the results of all the algorithms on the datasets. Give a "grade" between 1 and 5 to each to compare them:
  1. For example: Ward: [1, 2, 2, 1, 5, 3]
  2. Mini-batch KMeans: []
  3. Affinity propagation: []
  4. Mean shift: []
  5. Spectral clustering: []
  6. Ward: []
  7. Agglomerative clustering: []
  8. DB Scan: []
  9. Optics: []
  10. Birch: []
  11. Gaussian mixture: []
3. _QUESTION:_ Choose the most optimal algorithm to solve each dataset or situation and justify your answer:
  1. Dataset 1:
  2. Dataset 2:
  3. Dataset 3:
  4. Dataset 4:
  5. Dataset 5:
  6. Dataset 6:

**Demonstration of the assumptions of the K-means algorithm**

1. _QUESTION:_ Describe this dataset ("Unevenly sized blobs"), just as you described the datasets in the previous question:
  1. Answer:
2. _QUESTION:_ For each dataset or instance, describe the KMeans assumption that you think it fulfils or fails to fulfil in order to apply it correctly.
  1. Dataset 1:
  2. Dataset 2:
  3. Dataset 3:
  4. Dataset 4:
3. _QUESTION:_ For each dataset or instance, can you think of any possible transformation of the data to another variable space or any other type of transformation that would help us to solve such a problem with KMeans?
  1. Note: It's a complex question, so don't worry if you don't have an answer for each case, or can't think of anything, although we encourage you to try :).
  2. Dataset 1:
  3. Dataset 2:
  4. Dataset 3:
  5. Dataset 4:
