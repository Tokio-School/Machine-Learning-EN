# **SVM: Gaussian distribution**

M2U5 - Exercise 10

**What are we going to do?**

- We will implement the probability density function of a multivariate Gaussian/normal distribution
- Plot them graphically
- We will check how its behaviour varies by modifying its mean and covariance parameters

Remember to follow the instructions for the submission of assignments indicated in [Submission Instructions](https://github.com/Tokio-School/Machine-Learning/blob/main/Instrucciones%20entregas.md).

**Instructions**

The normal or Gaussian distribution is used in ML in models such as Gaussian kernel SVMs or anomaly detection, which we will see in a later module.

The distribution can be univariate (1 single characteristic) or multivariate (2 or more), which we will use in this exercise.

This distribution is defined by 2 parameters:

- The mean _mu_ of the features, a vector of size _n_.
- The _Sigma_ covariance matrix between the features, a 2D vector of size _n x n_.

In [ ]:

**import** numpy **as** np

**from** scipy.stats **import** multivariate\_normal

**from** matplotlib **import** pyplot **as** plt

**from** matplotlib **import** cm

**from** mpl\_toolkits.mplot3d **import** Axes3D

**Implement the probability density function of a Gaussian distribution**

This _PDF_ can be implemented with the SciPy method [scipy.stats.multivariate\_normal](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.multivariate_normal.html):

In [ ]:

_# TODO: Define mu and sigma ndarrays for 2 features or dimensions_

_# Define the mean as a 1D ndarray with any 2 values_

mu **=** [**...**]

_# Define Sigma as a 2D ndarray with any 2x2 values_

sigma **=** [**...**]

**Plot it graphically within a range of possible values**

To do this, create a linear space with Numpy of 100 values in the range [-5, 5] as an example:

In [ ]:

_# TODO: Calculate the PDF over a 2D linear space_

_# Create a meshgrid with this linear space: 100 values between [-5, 5]_

x1 **=** np **.** linspace([**...**])

x2 **=** np **.** linspace([**...**])

x1, x2 **=** np **.** meshgrid(x1, x2)

xy **=** np **.** empty(x1 **.** shape **+** (2,))

xy[:,:,0] **=** x1

xy[:,:,1] **=** x2

_# Calculate the PDF for this XY meshgrid, with the previously established mu and sigma_

z **=** multivariate\_normal([**...**])

In [ ]:

_# TODO: Plot the PDF in 3D on the space in question_

fig **=** plt **.** figure(1)

ax **=** fig **.** gca(projection **=**'3d')

ax **.** plot\_surface(x1, x2, z, rstride **=** 3, cstride **=** 3, linewidth **=** 1, antialiased **=True** , cmap **=** cm **.** viridis)

cset **=** ax **.** contour(x1, x2, z, zdir **=**'z', offset **=-** 0.5, cmap **=** cm **.** viridis)

ax **.** set\_zlim( **-** 0.15, 0.2)

ax **.** set\_zticks(np **.** linspace(0, 0.2, 5))

ax **.** view\_init(27, **-** 21)

plt **.** show()

_Bonus: Feel like experimenting?_ You can modify the parameters of the above representation to see how it affects the Matplotlib 3D plot.

**Check how its behaviour varies by modifying its parameters**

We have represented the PDF of the Gaussian distribution in 3D, using its 2 parameters, _mu_ and _Sigma._

_Why don't you check what happens when you change mu and Sigma? How does the graph vary when you change the 2 values of mu, separately, and at the same time? How does it vary when you change the 4 values of Sigma?_

Answer these questions with your conclusions:

- Varying _mu_...
- Varying _Sigma_...
