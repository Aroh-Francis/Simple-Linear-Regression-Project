Simple Linear Regression Project
This project demonstrates the application of simple linear regression using Python. It covers data visualization, preprocessing, model training, evaluation, and insights using both Scikit-learn and Stats models.

Project Features

•	Exploratory Data Analysis (EDA)

o	Scatter plot and correlation analysis.
o	Pair-plot visualization using Seaborn.

•	Data Preprocessing

o	Train-test split.
o	Standardization of features.

•	Simple Linear Regression

o	Model training with Scikit-learn.
o	Coefficients, intercept, and regression line visualization.
o	Performance metrics: MAE, MSE, RMSE, R², and Adjusted R².

•	OLS Linear Regression (Stats models)

o	Detailed statistical summary of the regression model.
o	Comparison with Scikit-learn predictions.

•	Predictions

o	Predictions for test data.
o	Predictions for new input data.


Project Workflow

1.	Load Data
o	Dataset is loaded from a CSV file (data.csv).

2.	Data Visualization
o	Scatter plot of Weight vs Height.
o	Correlation matrix.
o	Pair plot to visualize pairwise relationships.

3.	Preprocessing
o	Split data into training and test sets using train_test_split.
o	Standardize features using StandardScaler.

4.	Train and Evaluate Linear Regression Model
o	Train a linear regression model using Scikit-learn.
o	Plot the regression line and calculate performance metrics.

5.	OLS Regression for Detailed Analysis
o	Fit an Ordinary Least Squares (OLS) model using Stats models.
o	Display statistical summary (e.g., coefficients, p-values, R², etc.).

6.	Performance Metrics
o	Compute metrics such as MAE, MSE, RMSE, R², and Adjusted R².

7.	Make Predictions
o	Predict Height for new values of Weight.


How to Run the Project

1.	Clone the repository and ensure data.csv is present in the project directory.

2.	Install the required libraries:

pip install pandas NumPy matplotlib seaborn scikit-learn stat models

3.	Run the script:
python main.py

4.	Review the outputs in the terminal and the visualizations generated.
Key Outputs
•	Coefficients and Intercept: Understand the slope and intercept of the regression line.
•	Performance Metrics: Evaluate the model's accuracy using MAE, MSE, RMSE, R², and Adjusted R².
•	OLS Summary: Gain detailed insights into the statistical significance and goodness of fit.
•	Predictions: Get predictions for both test data and new data points.

Example Outputs
•	Coefficients: [1.26]
•	Intercept: 156.47
•	MAE: 3.21
•	R² Score: 0.85
•	Adjusted R² Score: 0.84

Future Improvements
•	Add more visualizations for deeper insights.
•	Explore polynomial regression for non-linear relationships.
•	Include cross-validation for robust evaluation.

Dependencies
•	Python 3.x
•	Libraries: pandas, NumPy, matplotlib, seaborn, sci-kit-learn, stats models

