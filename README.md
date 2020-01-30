# ml-code-python
# Data pre-processing

# Step 1: Importing the libraries

import pandas as pd
import numpy as np

# Step 2: Importing dataset

dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 3].values

# Step 3: Handling the missing data

# from sklearn.impute import SimpleImputer
# imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
# instead of
# from sklearn.preprocessing import Imputer
# imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
# DeprecationWarning: Class Imputer is deprecated; Imputer was deprecated in version 0.20 and will be removed in 0.22. Import impute.SimpleImputer from sklearn instead.

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

print(X)
print(Y)

# Step 4: Encoding categorical data

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[ : , 0] = labelencoder_X.fit_transform(X[ : , 0])

print(X)

# Creating a dummy variable

# onehotencoder = OneHotEncoder(categorical_features = [0])
# X = onehotencoder.fit_transform(X).toarray()

from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([("Country", OneHotEncoder(),[0])], remainder='passthrough')
# The last arg ([0]) is the list of columns you want to transform in this step
X = ct.fit_transform(X)

print(X)

labelencoder_Y = LabelEncoder()
Y =  labelencoder_Y.fit_transform(Y)

print(Y)

# Step 5: Splitting the datasets into training sets and Test sets

#from sklearn.cross_validation import train_test_split

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split( X , Y , test_size = 0.2, random_state = 0)

print(X_train)
print(X_test)

# Step 6: Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)

print(X_train)
print(X_test)
