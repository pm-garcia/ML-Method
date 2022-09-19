# Linear regression and gradient descent, manual implementation -> No use of frameworks

# Paul Martín García Morfín | A01750164
# Tecnológico de Monterrey 2022

'''
Used dataset: Boston Housing
This dataset contains information collected by the U.S Census Service concerning housing in the area of Boston Mass.
* The medv variable is the target variable (median value of a home).
Variables
    1. There are 14 attributes in each case of the dataset. They are:
    2. CRIM - per capita crime rate by town
    3. ZN - proportion of residential land zoned for lots over 25,000 sq.ft.
    4. INDUS - proportion of non-retail business acres per town.
    5. CHAS - Charles River dummy variable (1 if tract bounds river; 0 otherwise)
    6. NOX - nitric oxides concentration (parts per 10 million)
    7. RM - average number of rooms per dwelling
    8. AGE - proportion of owner-occupied units built prior to 1940
    9. DIS - weighted distances to five Boston employment centres
    10. RAD - index of accessibility to radial highways
    11. TAX - full-value property-tax rate per $10,000
    12. PTRATIO - pupil-teacher ratio by town
    13. B - 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
    14. LSTAT - % lower status of the population
    15. MEDV - Median value of owner-occupied homes in $1000's
* In this case, the variable LSTAT is used to make a simple linear regression model.

With the parameters used, the line equation found is:  34.73349627077651 + -0.9683693152536808 x
The metrics used for model evaluation are the MSE and RMSE: 
    - MSE: 40.96
    - RMSE: 6.39

At the end, the graphs of the found line are shown, as well as a dataframe comparing the actual y with the estimated y. 
'''

# Used libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Model for simple linear regression
def model(B0, B1, x):
    return B0+B1*x

# Error function: Mean Square Error
def MSE(y, y_):
    n = y.shape[0]
    mse = np.sum((y-y_)**2)/n
    return mse

# Gradient descent to minimize error function
def gradient_descent(B0_, B1_, lr, x, y):
    n = x.shape[0]
    # Error function derivatives
    B0_d = -(2/n)*np.sum(y-(B0_+B1_*x))
    B1_d = -(2/n)*np.sum((y-(B0_+B1_*x))*x)
    # Updating parameters with a learning rate
    B0 = B0_-lr*B0_d
    B1 = B1_-lr*B1_d
    return B0, B1

# Reading the data set
columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
housing = pd.read_csv('D:/paulm/Documents/Python Scripts/Universidad/IA y ciencia de datos I/Machine Learning/housing.csv', delim_whitespace=' ', names=columns)
print('The objective is to predict the median value of a home.')

# Correlation matrix
plt.figure(figsize=(10, 10))
sns.heatmap(housing.corr(), cmap='RdYlBu', 
    annot=True, square=True,
    vmin=-1, vmax=1, fmt='+.3f')
plt.title('Correlation matrix')
plt.show()
corr = housing.corr()
corr.drop(['MEDV'], axis=0, inplace=True)
maxvar = abs(corr[['MEDV']]).idxmax()[0]
print('Variable with the highest correlation factor: ', maxvar)

# Split the data into training (80%) and test (20%)
train = housing.sample(frac=0.8, random_state=25)
test = housing.drop(train.index)

# Using the LSTAT variable because it has a higher correlation factor 
x_train = train['LSTAT'].values
y_train = train['MEDV'].values
x_test = test['LSTAT'].values
y_test = test['MEDV'].values

# Graphing the data
plt.scatter(x_train, y_train)
plt.title('Scatter plot: Lower status VS Mean value')
plt.xlabel('Lower status')
plt.ylabel('Mean value')
plt.show()

# Initialize the parameters with a random value
np.random.seed(2)
B0 = np.random.randn(1)[0]
B1 = np.random.randn(1)[0]

# Define the learning rate and the number of iterations
lr = 0.0004
epochs = 40000

# Training
error = np.zeros((epochs, 1))
for i in range(epochs):
    [B0, B1] = gradient_descent(B0, B1, lr, x_train, y_train)
    y_ = model(B0, B1, x_train)
    error[i] = MSE(y_train, y_)

    # Print results every 1000 epochs
    if (i+1)%1000 == 0:
        print('Epoch {}'.format(i+1))
        print('    B0: {:.1f}'.format(B0), ' B1: {:.1f}'.format(B1))
        print('    MSE: {}'.format(error[i]))
        print('========================================')

# Graphing the error
plt.plot(range(epochs), error)
plt.title('MSE vs Epochs')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.show()

# Graphing the linear regression obtained
y_regr = model(B0, B1, x_train)
plt.scatter(x_train, y_train)
plt.plot(x_train, y_regr, 'r')
plt.title('Best-fit line')
plt.xlabel('Lower status')
plt.ylabel('Mean value')
plt.show()

# Making predictions and evaluating the model
y_pred = model(B0, B1, x_test)
y_mean = y_test.mean()
#ssreg = np.sum((y_pred-y_mean)**2)
#sstot = np.sum((y_test-y_mean)**2)
#r2_score = ssreg/sstot
mse = MSE(y_test, y_pred)
rmse = np.sqrt(mse)
y_comp = pd.DataFrame()
y_comp['y_test'] = y_test
y_comp['y_pred'] = y_pred
print('Value comparison')
print(y_comp)
print('The line equation found is: ', B0, '+', B1, 'x')
print('MSE for model evaluation: {}'.format(mse))
print('RMSE for model evaluation: {}'.format(rmse))
#print('R2 score for model evaluation: {}'.format(r2_score))
print('-'*50)