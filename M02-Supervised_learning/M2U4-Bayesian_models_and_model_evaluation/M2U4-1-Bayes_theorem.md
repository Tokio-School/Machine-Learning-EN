# **Bayes' Theorem: Sensitivity and Specificity**

M2U4 - Exercise 1

**What are we going to do?**

- We will implement the computation of probabilities using Bayes' Theorem in Python
- We will conduct several experiments to test sensitivity and specificity effects

Remember to follow the instructions for the submission of assignments indicated in [Submission Instructions](https://github.com/Tokio-School/Machine-Learning/blob/main/Instrucciones%20entregas.md).

**Instructions**

This will be a short and quick exercise, in which we will not introduce any new concepts.

In the course session, the slides, and the student manual we have solved a few exercises on Bayes' Theorem to get an idea of how it behaves, and how it can differ from our intuition.

We will now implement it programmatically to have the opportunity to repeat more of these examples, solving the calculations with Python, and graphing the results.

**Implement Bayes' Theorem in a function**

Let's start by implementing Bayes' Theorem with a Python function.

To do this, fill in the code in the following cell:

In [ ]:

_# TODO: Implement Bayes' Theorem in a function_

**def** bayes\_theorem(p\_a, p\_b\_given\_a, p\_b\_given\_not\_a):

""" Returns the probability of B given A according to Bayes' Theorem

All variables are percentages in float format (i.e., 10% -\> 0.1).

Positional arguments:

p\_a -- a priori probability

p\_b\_given\_a -- probability of occurrence of event B given event A

p\_b\_given\_not\_a -- probability that event B will not occur given event A

Return:

p\_a\_given\_b -- probability of occurrence of event A given event B

"""

**pass**

**return** p\_a\_given\_b

**Solve several examples and check the results**

To test your implementation, check it against the already solved examples in the slides:

Example 1:P(C)=0.01P(+|C)=0.9P(+|!C)=0.2P(C|+)=0.0435

Example 2:P(C)=0.01P(+|C)=0.85P(+|!C)=0.15P(C|+)=0.0541

Example 3:P(C)=0.01P(+|C)=0.9P(+|!C)=0.05P(C|+)=0.1538

Copy the code cell and display the 3 results:

In [ ]:

_# TODO: Solve the previous examples_

p\_a **=** 0.01

p\_b\_given\_a **=** 0.9

p\_b\_given\_not\_a **=** 0.2

p\_a\_given\_b **=** [**...**]

print('a priori probability of cancer:', p\_a)

print('Test sensitivity:', p\_b\_given\_a)

print('Test specificity:', p\_b\_given\_not\_a)

print('Probability of cancer if test positive:', p\_a\_given\_b)

**Plot the sensitivity and specificity effects graphically**

As we have seen, the a posteriori probability varied greatly depending on the sensitivity and specificity chosen.

To conclude the exercise, plot how the a posteriori probability varies as a function of sensitivity on one graph, and as a function of specificity on the other.

For this purpose, it establishes constant specificity in the first graph and constant sensitivity in the second graph:

In [ ]:

_# TODO: Plot the sensitivity and specificity effects in Bayes' theorem graphically_

p\_a **=** 0.01

_# Create an array with a linear spacing of +/- 25% over the base values of 0.9 and 0.2, respectively, with 10 values_

sensitivity **=** [**...**]

specificity **=** [**...**]

_# Calculate the results of the a posteriori probabilities_

_# Keep the central value of sensitivity and specificity constant, respectively_

res1 **=** [**...**]

res2 **=** [**...**]

fig, axs **=** plt **.** subplot(2)

plt **.** title()

axs[0] **=** plt **.** plot(res1)

axs[1] **=** plt **.** plot(res2)

plt **.** show()
