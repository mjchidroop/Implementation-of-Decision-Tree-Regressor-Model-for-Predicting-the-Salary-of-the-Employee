# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Feature Selection for Splitting: The algorithm chooses position levels as split points, seeking those that best minimize the error in salary prediction.

2. Recursive Partitioning: It divides the data into subsets, fitting constant salary values within each leaf node (terminal segment).

3. Stopping Criteria: Splitting continues until each segment is sufficiently small, pure, or other stopping conditions are met (e.g., minimum samples per leaf or a maximum tree depth).

4. Prediction: For new position levels, the tree is traversed to the appropriate leaf node, outputting the associated salary.

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: CHIDROOP M J
RegisterNumber:  25018548
*/

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

# Load dataset
data = pd.read_csv('Salary.csv')

# Use 'Level' to predict 'Salary'
X = data[['Level']]
y = data['Salary']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit Decision Tree Regressor
regressor = DecisionTreeRegressor(random_state=42)
regressor.fit(X_train, y_train)

# Predict and output results
y_pred = regressor.predict(X_test)
print('Model Score:', regressor.score(X_test, y_test))

X_grid = np.arange(min(X['Level']), max(X['Level']), 0.01).reshape(-1, 1)
y_grid_pred = regressor.predict(X_grid)

plt.figure(figsize=(8, 5))
plt.scatter(X, y, color='red', label='Actual Data')
plt.plot(X_grid, y_grid_pred, color='blue', label='Decision Tree Prediction')
plt.title('Decision Tree Regression: Level vs Salary')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


```

## Output:
<img width="293" height="67" alt="Screenshot 2025-10-05 155757" src="https://github.com/user-attachments/assets/9bfb43f2-f1f1-424d-9e8d-61ad5091fa87" />
<img width="1000" height="617" alt="Screenshot 2025-10-05 155745" src="https://github.com/user-attachments/assets/8ea7b2ba-28b8-4e70-9129-7d3e260e30a5" />



## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
