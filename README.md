# Nba-Draft-Prediction

This project aims to predict NBA draft picks using college data.
The data is collected from basketball-reference.com and sportsipy.
The data is cleaned and combined with the target value i.e. Win Shares.
The data is then split into train and test sets.
The train set is used to train the model and the test set is used to evaluate the model.
The model is a decision tree regressor.
The model is tuned using grid search.
The model is evaluated using mean squared error.

The project is divided into 2 parts:
1. Data Collection
2. Model Training and Evaluation


#### 1. Data Collection
The data is collected from basketball-reference.com and sportsipy.
You can use data_collection.py to collect the data. data_preparation() function can be used to collect and clean the data.
Then you can use get_proper_data() function to get the data in the proper format, for model training and evaluation.

#### 2. Model Training and Evaluation
There are 2 models in this project.
- Decision Tree Regression
- Random Forest Regression
The year range for training and testing data sets can be modified. Using earlier years for training and later years for testing is recommended.
The target value can be modified. The default target value is Win Shares.

The data is analyzed using pandas and seaborn.
Models plot their results to files to be used in the report. Pair plots for each feature are plotted as well.

Example result plots can be found in the report and plots file.
