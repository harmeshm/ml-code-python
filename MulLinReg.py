# Step 1: Data Preprocessing
# Importing the libraries
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 4].values
# print(dataset)
# print(X)
# print(Y)

# Encoding Categorical data

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

ct = ColumnTransformer([("State", OneHotEncoder(), [3])], remainder='passthrough')
# The last arg ([3]) is the list of columns you want to transform in this step
X = ct.fit_transform(X)
print(X)

# Avoiding Dummy Variable Trap
X = X[:, 1:]
print(X)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# Step 2: Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X_train, Y_train)

# Step 3: Predicting the Test set results
y_pred = regressor.predict(X_test)
# print(Y_test, y_pred)
