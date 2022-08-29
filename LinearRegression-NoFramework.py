# Linear regression, manual implementation - no use of frameworks

# Paul Martín García Morfín | A01750164
# Tecnológico de Monterrey 2022

# Used libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

# Split the data into training (80%) and test (20%)
train = housing.sample(frac=0.8, random_state=25)
test = housing.drop(train.index)
# Using the RM variable because it has a higher correlation factor 
x_train = train['RM'].values
y_train = train['MEDV'].values
x_test = test['RM'].values
y_test = test['MEDV'].values

# Graphing the data
plt.scatter(x_train, y_train)
plt.xlabel('Total rooms')
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
plt.title('Simple linear regression')
plt.xlabel('Total rooms')
plt.ylabel('Mean value')
plt.show()

# Making predictions and evaluating the model
y_pred = model(B0, B1, x_test)
y_mean = y_test.mean()
#ssreg = np.sum((y_pred-y_mean)**2)
#sstot = np.sum((y_test-y_mean)**2)
#r2_score = ssreg/sstot
mse = MSE(y_test, y_pred)
y_comp = pd.DataFrame()
y_comp['y_test'] = y_test
y_comp['y_pred'] = y_pred
print('Value comparison')
print(y_comp)
print('MSE for model evaluation: {}'.format(mse))
#print('R2 score for model evaluation: {}'.format(r2_score))