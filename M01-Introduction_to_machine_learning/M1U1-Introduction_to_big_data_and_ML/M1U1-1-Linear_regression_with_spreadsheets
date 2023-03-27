Linear Regression with Spreadsheets
M1U1 - Exercise 1
What are we going to do?
•	Resolve Linear Regression or least squares line fitting problems
•	We will implement the line fitting formulas in a spreadsheet
•	Graphically represent the initial data and the results
Remember to follow the instructions for the submission of assignments indicated in Submission Instructions.
Instructions
We are going to simulate a mathematical model, with a set of simulated data, to better understand how to perform a linear regression using, in this case, only any spreadsheet software, such as LibreOffice Calc, Microsoft Excel, Google Sheets, etc.:
•	Use any of these software, whichever one you are most comfortable with using their different sections. In particular, LibreOffice is a free office suite, and Google Sheets is free for all Gmail users.
•	Since each student may use a different program, we cannot explain the steps for any of these, except in general.
•	Remember that if you have any problem, you can quickly find the solution in any web search engine, or we will be happy to help you through the platform's messages.
•	In the same repository folder as this file, you will find several CSV files with different datasets. Depending on the instructions, for each task we will use one dataset or another.
•	You can work with several documents, one per task, or better, with several sheets in the same spreadsheet.
•	Call each sheet or document “task-1, task-2, etc.
Please, number the documents, or the sheets inside of just one document, descriptively, making clear which task each of them solves.
Example: model and data
In this example we are going to simulate an economic model: a price curve with the price of a product against its number of sales.
Mathematical/statistical/scientific models are used, regularly, to:
•	Explain the relation between 2 or more variables: the variable dependent on the rest (Y), and the independent variables (X), or model their behaviour.
•	From that model, future predictions are made, either interpolations or extrapolations.
We will be discussing both objectives during the exercise.
Least squares linear regression model
We are going to perform a least squares linear regression on our data. This linear regression will model the data, i.e., create a mathematical model from the data.
The set of data, also called a “dataset” in English, will contain our data. We have examined how the sale price of a product is influenced by its number of sales, through various experiments: different variations of the product, discounts and advertising campaigns, different markets, private sales etc.
The mathematical model will relate two variables:
•	The Y variable will be the number of sales, the object or dependent variable.
•	The X variable will be the price of the product, the independent variable.
We have simplified the names to X and Y to simplify the exercise.
Task 1
Data to be used: M1U1-2-dataset_tarea1.csv
Step 1
Before training any model, we should always try to visualise the data. In a model with a much more complete and complex dataset, it’s usually more complicated, but in a simple linear regression, it is very simple.
Create a graph of the points (or a “dot plot” or “scatter plot”) and show on the horizontal or X axis the X variable, and on the vertical or Y axis the Y variable.
In some cell of the document/sheet indicate “Question1” and answer the question:
What shape do you believe the resulting graph will follow? Why do you believe we speak, in this case, of linear regression?
Step 2
We are going to “model the data” (as they say in statistics) or “train the model” (as they say in machine learning).
For this, calculate the values of m and b according to the following formulas:
Y=m×X+bm=∑XY−(∑X)(∑Y)n∑X2−(∑X)2nb=Y―−m×X―
Notes:
•	The variables with a horizontal bar above indicate the average or “mean” of the variable, and the symbol Σ indicates summation, or to sum all the values of the variable.
•	Label the cells that will contain the values of m and b in a descriptive way, in order to monitor the exercise.
•	You can use the built-in functions of your spreadsheet software to find averages and summations or do them manually.
•	I recommend using other auxiliary cells/columns to calculate intermediate values when necessary.
With these coefficients, we have defined our model. These coefficients m and b are the ones that allow us to explain the behaviour of the variables and/or to make predictions with the model.
Step 3
We will evaluate the model using the correlation coefficient.
Remember that you can calculate it with the formulas:
R2=σXYσX⋅σY;σXY=XY―−X¯×Y¯;σX=∑X2n−X¯2;σY=∑Y2n−Y¯2
Notes:
•	Clearly indicate in a cell the value of R2.
•	To calculate X x Y, X2 or Y2, you can create auxiliary columns from the original columns, multiplying their values or squaring them.
•	You can check your calculations with the standard deviation and covariance functions of your spreadsheet software (be careful: use the standard deviation function for the whole population, not for a sample, as they may be different functions).
In some cell of the document/sheet indicate “Question 2 “ and answer the question:
What does this R2 value represent?
Step 4
Let's calculate the values of Y that the model would predict for each value of X, according to these values of m and b.
To do this, create another new column called y_pred and calculate the values for it according to the following formula:
Y=m×X+b
In some cell of the document/sheet indicate “Question 3” and answer the question:
What is the relationship between the results in that column and the value of R2?
Step 5
This step is optional
In your spreadsheet software, go back to your plot, add a trend line and calculate its R2 directly using the software's own functionality, which is usually available the in point plots.
Task 2
Data to be used: M1U1-2-dataset_task2.csv
Remember to work on a different sheet or document for this task, importing the columns of the dataset to be used in each of the steps.
Step 1
So far, we have used simulated data without any error, with a perfect direct correlation, which is not usually the case in real life.
Repeat steps 1 to 4 of the previous task (create the graph, calculate m and b, calculate R2 and calculate y_pred), except answering the questions, for the data in the columns X_real and Y_real.
Incorporate the y_pred column as a new series on the original graph of X_real and Y_real. If possible, incorporate this series as a line chart instead of points, to make it easier to visualise the trend line.
Step 2
Now let's calculate the residuals. The residuals are the absolute difference between Y real and y_pred. Calculate them for each value of X and then create a new graph where you plot the residuals on the vertical axis and X on the horizontal axis.
You will see that they are pseudo-random values that follow a normal distribution which we can call noise. These residuals in our dataset would correspond to errors in the measurements of the variables, random differences, hidden variables that were not taken into account, etc.
Step 3
On the other hand, we will use this new model to make predictions about 2 new types of values:
•	Interpolation is making predictions on values in the same range as the original dataset, between the maximum and minimum value.
•	Extrapolation is making predictions on values outside the range of the original dataset, with values below the minimum or above the maximum.
To do this, choose any 6 values, 3 of them within the range of the original dataset and another 3 outside that range, and predict their y_pred values.
Task 3:
Data to be used: M1U1-2-dataset_tarea3.csv
Step 1
Let's repeat the steps of the previous task with a new dataset, using columns X_error and Y_error.
For this data, create the plot, calculate m and b, calculate R2 and y_pred, add the trend line of y_pred to the plot and create the residuals plot.
In this case, analysing not only R2 but also the results and the graphs, we can see that our model is much worse than the previous ones.
Would you be able to figure out why the model doesn't work as well beforehand, just by looking at the dot plot with X_error and Y_error?
Step 2
Analyse the graph of X_error and Y_error:
What relationship would you say the original data has? Is it a linear relationship, or some other type of relationship?
Step 3
Can you think of any way to transform the original data so that it can be modelled using simple linear regression?
Hints:
•	Create a new column from the X of the original data.
•	Analyse the dot plot of X vs Y in detail.
•	Transform the original data in some way, e.g., by adding a value to it, multiplying it by a value, raising it to a number, or passing it through some function, etc.
•	The answer has 4 equal sides 8-).
Task 4
Data to be used: M1U1-2-dataset_tarea4.csv
On this occasion, load the data X_rand and Y_rand and plot them on a graph.
In some cell of the document/sheet indicate “Question 4 “ and answer the question:
Do you think we can train a model that finds some kind of linear relationship between the two variables, even by transforming the data? Why?
Task 5
In this task we are not going to use any dataset, but you are going to create one yourself.
We will simulate synthetic data that follows a certain relationship, in order to generate test datasets and check our algorithms and machine learning implementations.
To do this, create a new sheet or document where you are going to create several pairs of columns, according to the following instructions:
Step 1
We are going to generate a dataset similar to the one we used in the first task. To do this, follow these steps:
1.	Create a column X_step1, with values in the range [0, 10, 20, …, 100].
2.	Create 2 cells for your m and b values and give them any 2 values.
3.	Generate another calculated column Y_step1 with your values of m, b, and X_step1.
Step 2
Now we are going to generate a dataset with random noise.
To do this, follow the instructions in the previous step to generate a pair of columns X_step2 and Y_step2, and in addition:
1.	We are going to add a noise term to the column Y_step2.
2.	Create a new auxiliary cell e that will represent the error scale. It can be in units, tens, hundreds, decimals, even negative numbers.
3.	Create a new columnY_step2_error with the formula Y_step2_error=Y_step2+Y_step2×N×e.
4.	N is a random number in the range [-1, 1] that we can generate using the random number generation function (normal distribution or not) of your spreadsheet software following the formula: (RANDN - 0.5) * 2
5.	Plot the values of the columns X_step2, Y_step2 and Y_step2_error on a dot plot.
6.	Play around by varying the value of e, seeing how it affects the values of Y_step2_error, until you get a more or less normal error.
Step 3
Now we are going to generate a dataset with polynomial data.
To do this, follow the steps in Step 1 to generate a pair of columns X_step3 and Y_step3, and in addition:
1.	If you prefer, you can use a new sheet or document.
2.	Create a new column X_step3_squared by squaring the values of X_step3.
3.	Change the Y_step3 column to use the steps in the X_step3_squared column instead of the X_step3 column.
4.	Plot both columns X_step3_squared and Y_step3 on a dot plot.
Step 4
Finally, we are going to create data with relationships other than a linear or polynomial relationship:
1.	If you prefer, you can use a new sheet or document.
2.	Create a column X_step4,with values in the range [0, 10, 20, …, 100].
3.	Create another column Y_step4 and calculate its values using the formula Y_step4 = 3 * log(X_step4 + 2)
4.	Plot both columns on a dot plot.
Task 6
Let's get to know each other! And see how we can contact you through the platform.
For this task, send me (instructor Marcos Manuel Ortega) a message through the platform with the following information:
1.	Look up information about the following sentence and analyse it. Explain to me why you think it is interesting to always remember it when analysing data. The phrase is: "Correlation does not imply causation".
2.	Find some articles, blogs, or similar documents where they show examples of statistical data analysis where correlation did not imply causation and send them to me.
3.	Find a meme, joke, or funny phrase related to that phrase or something similar and share it, There are many :D !
4.	Look for the famous XKCD comic strip about this phrase, analyse it, and send it to me along with an explanation of the joke.
We have learned that...
•	So far, we haven't done real machine learning. We have limited ourselves to creating statistical models, analysing and evaluating them.
•	We have seen, for now, how machine learning is nothing more than performing statistical modelling. During the course we will see that the difference is in the implementation and in using more advanced algorithms.
•	We have seen the need to analyse data prior to attempting to train any model.
•	We have discovered the concept of residuals.
•	We have seen how we often have to transform data before using it to train a model.
•	We have learned how to create our own synthetic datasets to test models and their implementation.


