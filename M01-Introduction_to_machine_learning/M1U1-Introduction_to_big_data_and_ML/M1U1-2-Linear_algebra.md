# **Linear Algebra**

M1U1 - Exercise 2

**What are we going to do?**

- Solving a series of linear algebra mathematical operations problems

Remember to follow the instructions for the submission of assignments indicated in [Submission Instructions](https://github.com/Tokio-School/Machine-Learning/blob/main/Instrucciones%20entregas.md).

**Instructions**

We will perform a series of linear algebra mathematical operations, to help us to fully internalise these concepts.

For this exercise you are asked to solve these exercises in the way you prefer: using a text editor capable of including mathematical formulas, LaTeX, a scanned document, or even a photograph of a sheet where the solution is clearly visible.

Please remember to solve them manually, step by step. Of course, you can use any calculator to check the result, but we want to set out the features of the operation in order for you to understand it clearly.

Once you have completed these operations manually, this will ensure that you will not have any problems performing them using code in more complex implementations, such as the ones we will do during the course.

_Note from the teacher: Although it may not sound very sexy to do a high school maths exercise, being able to solve it without any problem and internalising these concepts will help you avoid many problems during the course and in your work, in my experience with ML :)._

**Matrices and Vectors**

{
   "cell_type": "markdown",
   "id": "9407f0bb-63a3-41ca-9a1e-a7e5e9a0f7c6",
   "metadata": {},
   "source": [
    "### Matrices y vectores\n",
    "\n",
    "$A_{3\\times3} = \\begin{bmatrix}\n",
    "1 & 2 & 3 \\\\\n",
    "4 & 5 & 6 \\\\\n",
    "7 & 8 & 9 \\\\\n",
    "\\end{bmatrix}$\n",
    "\n",
    "$B_{3\\times3} = \\begin{bmatrix}\n",
    "10 & 20 & 30 \\\\\n",
    "40 & 50 & 60 \\\\\n",
    "70 & 80 & 90 \\\\\n",
    "\\end{bmatrix}$\n",
    "\n",
    "$V_{3\\times1} = \\begin{bmatrix}\n",
    "a \\\\\n",
    "b \\\\\n",
    "c \\\\\n",
    "\\end{bmatrix}$\n",
    "\n",
    "$C_{3\\times2} = \\begin{bmatrix}\n",
    "a & b \\\\\n",
    "c & d \\\\\n",
    "e & f \\\\\n",
    "\\end{bmatrix}$\n",
    "\n",
    "$D_{2\\times3} = \\begin{bmatrix}\n",
    "g & h & i \\\\\n",
    "j & k & l\n",
    "\\end{bmatrix}$\n",
    "\n",
    "$Y_{4\\times1} = \\begin{bmatrix}\n",
    "y_1 \\\\\n",
    "y_2 \\\\\n",
    "y_3 \\\\\n",
    "y_4\n",
    "\\end{bmatrix}$\n",
    "\n",
    "$\\Theta_{1\\times3} = \\begin{bmatrix}\n",
    "\\theta_1 & \\theta_2 & \\theta_3\n",
    "\\end{bmatrix}$\n",
    "\n",
    "$X_{4\\times3} = \\begin{bmatrix}\n",
    "1 & 2 & 3 \\\\\n",
    "4 & 5 & 6 \\\\\n",
    "7 & 8 & 9 \\\\\n",
    "10 & 11 & 12\n",
    "\\end{bmatrix}$"
   ]
  },

B3×3=[102030405060708090]

V3×1=[abc]

C3×2=[abcdef]

D2×3=[ghijkl]

Y4×1=[y1y2y3y4]

Θ1×3=[θ1θ2θ3]

X4×3=[123456789101112]

**Exercises**

1. Transpose matrix _A_
2. Find the sum of matrix _A_ and _B_.
3. Multiply matrix _A_ by the scalar number 3.
4. Multiply matrix _A_ by the vector _V_.
5. Multiply matrix _A_ by matrix _C_.
6. Multiply matrix _B_ by matrix _C_.
7. Multiply matrix _B_ by matrix _D_.
8. Given that in a linear regression _Y_ is equal to the product of the vector _Θ_ and the matrix _X_, state the equation and solve it for each element of _Y_, taking into account:
  1. _Y_ and _Theta_ are row or column vectors, we can transpose them without any problem.
  2. We can transpose _X_ only if there is no other option.
  3. We can choose the order of the multiplication as we wish.

**We have learned that...**

- The implementation of machine learning relies on linear algebra calculations.
- We need to know how to multiply matrices step by step in order to know how to utilise these operations when implementing machine learning algorithms.
- Not all operations are always possible.
- Be careful with the dimensions of vectors and matrices, _extremely careful!_