# **Final Project**

M06 - Exercise 1

**What are we going to do?**

- We will prepare the final project
- We will submit it for review and evaluation
- We will publish the project on our Kaggle profile

**General guidelines**

As a final project we will train a machine learning model that solves a problem and a dataset that we will choose ourselves.

_INSTRUCTIONS:_

1. Search for and choose a dataset for a use case you are interested in at [OpenML](https://www.openml.org/).
2. You can solve a dataset for supervised, semi-supervised, or unsupervised learning, for regression, classification, anomaly detection, etc., take your pick.
3. As throughout the course, use the development environment of your choice: local, WSL, VM, Google Colab, cloud, etc.
4. Start by creating a Jupyter notebook that will be your final deliverable.
5. Document the whole process in the notebook:
  1. Remember that the purpose of a notebook is not only to show the final result, but you can also include cells from each test, iteration, or hypothesis, to document what you have tested and what results they have yielded.
  2. Variable names for different tests, such as differently preprocessed datasets or different models, can have a different name so that you can document your work and go back to continue refining the model in a different way.
  3. You do not need to set up new variables or a new cell for each step you take, only for each main hypothesis, especially for each step whose results you then want to compare with other hypotheses or experiments.
  4. So, you don't need a new cell or new variables every time you correct a problem or change the value of a hyperparameter, but you do e.g., if you want to compare a decision tree versus a logistic regression, or if you want to compare your model on a base dataset versus several versions of a more processed dataset.
6. Work on the basis of hypotheses or experiments:
  1. Experiments are different steps that we propose, and depending on their results, we decide to move forward in one way or another.
  2. We can experiment with, among others, different feature sets, feature extraction or processing, model families, models and their hypeparameters, etc.
  3. Try whenever possible to follow the scientific method, data science:
    1. State and explain the original or current situation.
    2. Discuss what experiments you can propose to move forward and their justification.
    3. Outline the results of each of them
    4. Evaluate these results and justify your next step.
  4. Remember not to simply try all available options, but to justify why, and especially to check their suitability beforehand.
  5. In each case, document the corresponding pre- and post-hypothesis metrics, highlighting the difference.
7. The aim is to document your work to show your result and to show, broadly speaking, what steps you have taken, what you have tried and what results you have obtained, and why you have continued to move forward in one way or another.

**Define use case**

1. Start by presenting your chosen use case. Tell us a bit about why you are interested.
2. Find and reference some literature citations, examples or tutorials that reference the dataset or similar ones, and draw on such works whenever possible.

**Dataset analysis**

Introduce the dataset to be used and start by doing an exploratory data analysis of the dataset:

- Origin of the dataset, documentation, links and references (if possible).
- Size.
- Features.
- Preprocessing already done or "data lineage".
- Licence for the same.
- Will you have enough data for your use case, based on the required features? What number of features can you afford with that amount of data?
- To explore it, you can use Numpy or Pandas, Matplotlib or Seaborn, etc.

**Define the technical requirements**

An ML model is nothing more than an application, and an ML-based system is nothing more than a software solution, so our process must begin by defining the minimum technical requirements:

- Proposed model evaluation metrics.
- Minimum acceptable final accuracy of the model.
- Desired final model accuracy.
- Allocable resources (CPU and RAM).
- Maximum training time.
- Documenting dependencies (libraries like Numpy, Pandas, Scikit-learn, etc.)

**Exploratory data analysis**

We need to understand the data in detail to train our model, as the data will be its fuel.

Conduct an exploratory data analysis to dive into it. If you have taken the Python and AI course at Tokio School, you can build on your knowledge of Python for data science:

1. Present the dataset/dataframe: shape, features, feature types, dataset summary.
2. Check for incomplete, incorrect, invalid, null, non-uniform values, etc.
3. Feature analysis:
  1. Numerical: attributes such as mean, standard deviation, distribution, and representation as dot plot and boxplot.
  2. Categorical: ordinal or nominal, mode, number of categories, and representation as a histogram.
4. Define your Target variable.
5. Outlier analysis.
6. Clean incorrect, invalid data and assess whether or not to remove outliers and incomplete data.
7. Preprocess the features:
  1. Numerical: as such, bucketizing or categorising, polynomials, roots and logarithms, etc.
  2. Categorical: ordinal, one-hot encoding, leading to numeric.
  3. Possible feature crossovers.
8. If you prefer, fill in incomplete data with interpolations or averages.
9. Randomly reorder the data.
10. Rescale or normalise the training data.
11. Summarize the features and their attributes and distributions after preprocessing.
12. Find the correlation with the target variable with a correlation matrix plotted on a graph.
13. Divide your dataset into training, validation, and test subsets, or use K-fold.

**Auxiliary functions**

Define as many auxiliary code functions as you need to do repetitive tasks, such as representing variables, distributions, validating models, presenting metrics, etc.

**Base model**

Start your work by defining a base model:

1. The base model must be moderately adequate, not simply the first one we come up with, not in accuracy but in appropriateness to the envisaged case.
2. Evaluate the model:
  1. Plot its metrics.
  2. Check the residuals.
  3. Check for deviation or overfitting.
3. For each hypothesis or experiment, compare your results against each other and against the baseline model, to check whether you are improving or not.

**Feature engineering**

Start from the baseline data and conduct experiments or hypotheses to improve your features.

When you don't know how to move forward with your model, you can go back to the starting point and improve the features used instead:

1. Evaluate the relevance of features after training your models.
2. Propose some improvements, such as proposing polynomials of different degrees (no more than 4/5), logarithms, intersections between variables, etc.
3. Consider PCA or reduce the features if the dimensionality/complexity of the model is very high.
4. Incorporate new features.

**Refining the model**

Start planning and refining your models:

- Try different families of models related to your use case: decision trees, linear models, SVM/SVRs, etc.
- Try different types of models within families.
- Advance by posing hypotheses or experiments, evaluating and commenting on the results.
- Continue to improve your model iteratively, by setting out a few possibilities, exploring them, and moving on to the most promising one and going deeper into it.
- Evaluate each model or experimental approach:
  - Present each model as a new version, with a descriptive, versioned name.
  - Selected metrics.
  - Training time.
  - It suffers from deviation or over-fitting.
  - Comparison with other models in the same experiment and with the baseline model.
- Try more advanced methods if necessary:
  - Return to the feature extraction stage.
  - PCA and dimensionality reduction.
  - Optimization of hyperparameters: training rate, regularization, etc.
  - GridSearch, genetic algorithms.
  - Ensembles.

**Model presentation**

Present your final model

- Typology
- Features used
- Hyperparameters
- Metrics and results
- Training time and nº of iterations
- Nº of training examples
- Variance or overfitting
- Comparison with baseline model
- Display your weights/coefficients/parameters (including bias)
- Justification of their suitability and compliance with the initial technical requirements.

**Profile on Kaggle**

Upload your work to your Kaggle profile! This is how you can start your public portfolio.

Remember to clean up your notebook before uploading, removing references to the course where necessary.

**Presentation of the project**

Submit your project for evaluation.

Process:

1. Send the deliverables required by the platform.
2. The instructor will give you feedback, and if it is positive, will proceed to ask 3 questions for you to answer.
3. Record your project presentation, use case, and results, and your answers in a short video of less than 10 minutes, showing you and/or your screen.
4. Your final grade will be the combination of your feedback on the project and your presentation of the project.

_DELIVERABLES:_

1. Notebook named " M6-1-Final\_Project\_name\_of\_student.ipynb".
2. Recording with the answers to the 3 questions once they have been asked.
