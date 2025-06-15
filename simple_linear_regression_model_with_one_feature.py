import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import os
import logging
import sys

################################### CONSTANTS ####################################################

PATH_TO_DATA = "/home/umair/Documents/AI_TRAINING/Boston_housing/boston-housing-dataset/HousingData.csv"
# This is the path to the dataset file. Make sure it exists. It can be downloaded from:
# https://www.kaggle.com/datasets/altavish/boston-housing-dataset/data

##################################################################################################

################################### LOGGING ######################################################
logging.basicConfig(level=logging.INFO, format='%(levelname)s - Line %(lineno)d - %(message)s', handlers=[
    logging.StreamHandler(sys.stdout)
])

app_logger = logging.getLogger('SimpleLinearRegressionApp')
app_logger.setLevel(logging.INFO)
##################################################################################################

try:
    '''
    Load the Boston Housing dataset and perform simple linear regression using one feature (RM - average number of rooms per dwelling)
    The target variable is MEDV (median value of owner-occupied homes in $1000s).
    '''
    data = pd.read_csv(PATH_TO_DATA)

    # Display basic information about the dataset
    logging.info("############################################## LINEAR REGRESSION MODEL FOR BOSTON HOUSING DATASET ##############################################")
    logging.info(f"Dataset shape:{data.shape}")
    logging.info(f"Features:")
    logging.info(data.columns.tolist())
    logging.info(f"First few rows:")
    logging.info(data.head())

    # Check for missing values
    logging.info("\nMissing values per column:")
    logging.info(data.isna().sum())

    logging.info("\nMissing values will be dropped from the analysis:")

    # Drop rows that have NaN in AGE or MEDV
    data = data.dropna(subset=['RM', 'MEDV'])

    X = data[['RM']]
    y = data[['MEDV']]


    # Split the data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    logging.info(f"\nTraining set size: {X_train.shape[0]} samples")
    logging.info(f"Testing set size: {X_test.shape[0]} samples")

    # Train the linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions on test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    logging.info("\nModel Performance:")
    logging.info(f"Mean Squared Error: {mse:.2f}")
    logging.info(f"Root Mean Squared Error: {rmse:.2f}")
    logging.info(f"R² Score: {r2:.2f}")

    logging.info("############################################################################################")

    # Scatter Plot + Regression line
    plt.figure(figsize=(8, 6))
    plt.scatter(X_test, y_test, alpha=0.7, label='Actual', edgecolors='b')
    plt.plot(X_test, y_pred, color='red', label='Regression Line')  # predicted line
    plt.title('Linear Regression: Rooms vs MEDV')
    plt.xlabel('Rooms (RM)')
    plt.ylabel('MEDV (Median value of homes in $1000s)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

except FileNotFoundError as e:
    app_logger.error(f"File not found: {PATH_TO_DATA}. Please check the path and try again.")
    sys.exit(1)
except Exception as e:
    app_logger.error(f"An error occurred: {e}")
    sys.exit(1)
    
