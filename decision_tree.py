# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13, 2023
@author: erdem
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the data from the CSV file
df = pd.read_csv("data.csv", sep=";", header=None)

# Extract the independent variable (x) and dependent variable (y)
x = df.iloc[:, 0].values.reshape(-1, 1)
y = df.iloc[:, 1].values.reshape(-1, 1)

#%% Decision Tree Regression
from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor()   # random state = 0
tree_reg.fit(x, y)
predicted_value = tree_reg.predict([[5.5]])

x_new = np.arange(min(x), max(x), 0.01).reshape(-1, 1)  # Generate new data for testing
y_head = tree_reg.predict(x_new)

#%% Visualization
plt.scatter(x, y, color="red")
plt.plot(x_new, y_head, color="green")
plt.xlabel("Grandstand Level")
plt.ylabel("Grandstand Price")
plt.show()