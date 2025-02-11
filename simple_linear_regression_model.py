import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# import data
df = pd.read_csv('realest.csv')

# pre process the data (notice there are missing values ('NA') in many of the fields
print("Missing values before cleaning:\n", df.isnull().sum())
# We have a couple good ways to resolve this:

# Option 1: Drop rows with NaN values (but we're losing out on data points)
# df = df.dropna()

# Option 2: Fill missing values with the median of each column
imputer = SimpleImputer(strategy="median") # imputer fills in missing values using a chosen strategy
df.iloc[:, :] = imputer.fit_transform(df) # scans the dataframe (df), finds missing values, computes medians,  fills NaNs.

# select single feature for simple regression
X = df[['Space']]  # independent variable (size in sq ft)
y = df['Price']  # dependent variable (price)

# Split dataset into 75% training 25% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Train linear regression model
linear_reg = LinearRegression()
linear_reg.fit(X_train, y_train)


# Make predictions
y_pred = linear_reg.predict(X_test)
# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error: {mae:.2f}")
print(f"R-squared Score: {r2:.2f}")

# R squared score = coefficient of determination
# refers to the variability in dependent Y that is being explained by the independent variables Xi in the reg model.

# Plot the regression line
plt.figure(figsize=(8,6))
plt.scatter(X_train, y_train, color='blue', label="Training Data")
plt.scatter(X_test, y_test, color='red', label="Test Data")
plt.plot(X_test, y_pred, color='black', linewidth=2, label="Regression Line")
plt.xlabel("House Space (sq ft)")
plt.ylabel("House Price")
plt.title("Simple Linear Regression: House Price vs. Space")
plt.legend()
plt.show()

# Result example
# Mean Absolute Error: 8.54  <-- suggests moderate errors, model is not accurate (lower is better)
# On average, predictions are $8,540 off from the actual price (we assume the values in Price column are in thousands of dollars)

# R-squared Score: 0.38 (higher is better) <-- 38 percent suggests a weak relationship.
# R^2 measures how well the model explains price variation based on house space
# 0.38 means only 38% of the price variation is explained by the model.