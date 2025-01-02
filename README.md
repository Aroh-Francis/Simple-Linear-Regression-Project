Simple Linear Regression Project 

This project demonstrates the use of Simple Linear Regression to predict a dependent variable based on an independent feature using Python. The project follows the CRISP-DM methodology for data mining, coupled with the STAR technique for each level of the process.

Table of Contents
Project Overview
Data Understanding
Data Preparation
Modeling
Evaluation
Conclusion


Project Overview
The goal of this project is to develop a simple linear regression model to predict a person's height based on their weight. This project involves importing necessary libraries, loading the dataset, performing exploratory data analysis (EDA), building a predictive model, and evaluating the model's performance.

Data Understanding
Dataset:
The dataset is stored in a CSV file (data.csv), which contains two features: Weight and Height.
The first few rows of the dataset look like this:
Weight	Height
45	120
58	135
48	123
60	145
70	160
The task is to predict the Height (dependent variable) from the Weight (independent variable).


Data Preparation
Steps:
Loading the data: The dataset is loaded into a Pandas DataFrame using pd.read_csv().

Visualizing the data: A scatter plot is created to visualize the relationship between Weight and Height.

Checking correlations: The correlation between Weight and Height is computed using the .corr() method, which shows a strong correlation (0.931).

Seaborn pairplot: A pairplot is generated for a better understanding of the data distribution.

Feature selection:

X: Independent feature (Weight)
Y: Dependent feature (Height)
Train-test split: The data is split into training and testing sets using train_test_split() from scikit-learn. The training set is used to build the model, and the testing set is used for evaluation.

Standardization: The features are standardized using StandardScaler to ensure better model performance.


Modeling
Steps:
Linear Regression: A simple linear regression model is built using LinearRegression from scikit-learn. The model is trained on the training data (X_train, Y_train).

Model coefficients: The model outputs the coefficient (slope) and intercept of the regression line.

Training data visualization: A regression line is plotted alongside the training data.

Prediction on the test set: Predictions are made on the testing set (X_test).

Evaluation
Performance Metrics:
Mean Absolute Error (MAE): Measures the average magnitude of the errors.
Mean Squared Error (MSE): Measures the average of the squared errors.
Root Mean Squared Error (RMSE): Square root of the MSE, gives a sense of error magnitude.
R² Score: Measures the proportion of variance in the dependent variable that is predictable from the independent variable.
Adjusted R²: Adjusted for the number of features in the model.
The performance of the model is evaluated using the following metrics:

MAE: 9.67
MSE: 114.84
RMSE: 10.72
R² Score: 0.736
Adjusted R² Score: 0.670
Conclusion
The simple linear regression model predicts the height based on weight with reasonable accuracy. The R² score of 0.736 indicates that the model explains 73.6% of the variance in the data. The Adjusted R² score of 0.670 suggests that, even after accounting for the number of features, the model performs well.

Additional Information:
The model was also evaluated using OLS (Ordinary Least Squares) regression, which returned similar performance metrics to the linear regression model.
CRISP-DM and STAR Technique
The project follows the CRISP-DM (Cross-Industry Standard Process for Data Mining) methodology:

Business Understanding (STAR - Situation): The project aims to predict a person's height based on their weight.
Data Understanding (STAR - Task): The dataset contains two features: Weight and Height, with a strong correlation between them.
Data Preparation (STAR - Action): The dataset is cleaned, visualized, and preprocessed (train-test split, scaling).
Modeling (STAR - Result): A simple linear regression model is built, and the performance metrics are calculated.
Evaluation (STAR - Reflection): The model is evaluated using MAE, MSE, RMSE, R², and Adjusted R².
Requirements
Python 3.x
pandas
numpy
matplotlib
seaborn
scikit-learn
statsmodels
To install the dependencies, use the following:

bash
Copy code
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels
License
This project is open-source and available under the MIT License.






