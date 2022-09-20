# Random Forest implementation -> Use of RandomForestRegressor from Sklearn

# Paul Martín García Morfín | A01750164
# Tecnológico de Monterrey 2022

'''
Used dataset: Boston Housing
This dataset contains information collected by the U.S Census Service concerning housing in the area of Boston Mass.
* The medv variable is the target variable (median value of a home).
Attributes 
    1. CRIM - per capita crime rate by town
    2. ZN - proportion of residential land zoned for lots over 25,000 sq.ft.
    3. INDUS - proportion of non-retail business acres per town.
    4. CHAS - Charles River dummy variable (1 if tract bounds river; 0 otherwise)
    5. NOX - nitric oxides concentration (parts per 10 million)
    6. RM - average number of rooms per dwelling
    7. AGE - proportion of owner-occupied units built prior to 1940
    8. DIS - weighted distances to five Boston employment centres
    9. RAD - index of accessibility to radial highways
    10. TAX - full-value property-tax rate per $10,000
    11. PTRATIO - pupil-teacher ratio by town
    12. B - 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
    13. LSTAT - % lower status of the population
    14. MEDV - Median value of owner-occupied homes in $1000's
* In this case, all variables ares used to make a random forest model.

With the parameters used, the metrics used for the model are MSE and RMSE: 
    - MSE: 7.93
    - RMSE: 2.82
    - R2: 0.87

At the end, the graphs of the model are shown. 
'''

# Used libraries
import pandas               as pd
import numpy                as np 
import matplotlib.pyplot    as plt
import seaborn              as sns

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import GridSearchCV
import multiprocessing


# Reading the data set
columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
housing = pd.read_csv('D:/paulm/Documents/Python Scripts/Universidad/IA y ciencia de datos I/Machine Learning/housing.csv', delim_whitespace=' ', names=columns)
print('\nThe objective is to predict the median value of a home.\n')

# -- EDA --
print('-'*20, 'EDA', '-'*20)
print('\nDataset shape: ')
print(housing.shape)

print('\nSummary: ')
print(round(housing.describe().transpose(), 2))

print('\nNumerical variables: ')
for x in housing.select_dtypes(exclude=['object']):
    print('\t' + x)

print('\nCategorical variables: ')
for x in housing.select_dtypes(include=['object']):
    print('\t' + x)

print('\nMissing data count: ')
missing_data = housing.isna().sum()
print(missing_data[missing_data > 0])

print('\nDuplicated data count: ')
duplicated_data = housing.duplicated().sum()
print(duplicated_data[duplicated_data > 0])
print('-'*45)
# ---------

# Correlation matrix
plt.figure(figsize=(10, 10))
sns.heatmap(housing.corr(), cmap='RdYlBu', 
    annot=True, square=True,
    vmin=-1, vmax=1, fmt='+.3f')
plt.title('Correlation matrix')
plt.show()

# Variables (target and features) 
y = housing['MEDV']
X = housing.drop(['MEDV'], axis=1).copy()

# Data scaling
s_scaler = StandardScaler()
s_scaler.fit(X)
X_trans = s_scaler.fit_transform(X)

# Split the data into training (80%) and test (20%)
X_train, X_test, y_train, y_test = train_test_split(X_trans, y, test_size=0.2, random_state=42)

# Random forest model
forest = RandomForestRegressor(n_estimators = 150,
            criterion    = 'squared_error',
            max_depth    = 20,
            max_features = 5,
            oob_score    = True,
            n_jobs       = -1,
            random_state = 42)
forest.fit(X_train, y_train)

# Predictions
y_hat_train = forest.predict(X_train)
y_hat_test = forest.predict(X_test)

# Learning curve (Bias, Variance, Trade-off)
train_sizes, train_scores, test_scores = learning_curve(estimator=forest,
                        X=X_train, y=y_train, 
                        train_sizes=np.linspace(0.1, 1.0, 10), cv=10,
                        n_jobs=-1)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.plot(train_sizes, train_mean, color='r', marker='o', markersize=5,
         label='Train')
plt.fill_between(train_sizes, train_mean + train_std, 
                 train_mean - train_std, alpha=0.15, color='r')
plt.plot(train_sizes, test_mean, color='b', linestyle='--', 
         marker='s', markersize=5, label='Test')
plt.fill_between(train_sizes, test_mean + test_std, 
                 test_mean - test_std, alpha=0.15, color='b')
plt.grid()
plt.title('Learning Curve')
plt.legend(loc='upper right')
plt.xlabel('Number of training samples')
plt.ylabel('Accurancy')
plt.show()

# Y comparison 
y_comp = pd.DataFrame()
y_comp['y_test'] = y_test
y_comp['y_pred'] = y_hat_test

print('\nValue comparison')
print(y_comp)

# Model in train
mse_train = mean_squared_error(y_hat_train, y_train)
mae_train = mean_absolute_error(y_hat_train, y_train)
rmse_train = np.sqrt(mse_train)
r2_train = r2_score(y_hat_train, y_train)
print('\nThe model has the following metrics in train:')
print('MSE: ', mse_train)
print('RMSE: ', rmse_train)
print('MAE: ', mae_train)
print('R2: ', r2_train)

# Model in test
mse_test = mean_squared_error(y_hat_test, y_test)
mae_test = mean_absolute_error(y_hat_test, y_test)
r2_test = r2_score(y_hat_test, y_test)
rmse_test = np.sqrt(mse_test)
print('\nThe model has the following metrics in test:')
print('MSE: ', mse_test)
print('RMSE: ', rmse_test)
print('MAE: ', mae_test)
print('R2: ', r2_test)
print('')

# Hyperparameter optimization
# Grid search
param_grid = {'n_estimators': [150],
              'max_features': [5, 7, 9],
              'max_depth'   : [None, 3, 10, 20]
             }

# Cross-validation
grid = GridSearchCV(
        estimator  = RandomForestRegressor(random_state=42),
        param_grid = param_grid,
        scoring    = 'neg_root_mean_squared_error',
        n_jobs     = multiprocessing.cpu_count() - 1,
        cv         = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42), 
        refit      = True,
        verbose    = 0,
        return_train_score = True
       )

# New model
final_forest = RandomForestRegressor(n_estimators = 100,
            criterion    = 'squared_error',
            max_depth    = None,
            max_features = 1.0,
            oob_score    = False,
            n_jobs       = -1,
            random_state = 42)
final_forest.fit(X_train, y_train)

# New predictions
y_hat_train = final_forest.predict(X_train)
y_hat_test = final_forest.predict(X_test)

# Learning curve (Bias, Variance, Trade-off)
train_sizes, train_scores, test_scores = learning_curve(estimator=final_forest,
                        X=X_train, y=y_train, 
                        train_sizes=np.linspace(0.1, 1.0, 10), cv=10,
                        n_jobs=-1)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.plot(train_sizes, train_mean, color='r', marker='o', markersize=5,
         label='Train')
plt.fill_between(train_sizes, train_mean + train_std, 
                 train_mean - train_std, alpha=0.15, color='r')
plt.plot(train_sizes, test_mean, color='b', linestyle='--', 
         marker='s', markersize=5, label='Test')
plt.fill_between(train_sizes, test_mean + test_std, 
                 test_mean - test_std, alpha=0.15, color='b')
plt.grid()
plt.title('Learning Curve')
plt.legend(loc='upper right')
plt.xlabel('Number of training samples')
plt.ylabel('Accurancy')
plt.show()

# Y comparison 
y_comp = pd.DataFrame()
y_comp['y_test'] = y_test
y_comp['y_pred'] = y_hat_test

print('\nValue comparison')
print(y_comp)

# Model in train
mse_train = mean_squared_error(y_hat_train, y_train)
mae_train = mean_absolute_error(y_hat_train, y_train)
rmse_train = np.sqrt(mse_train)
r2_train = r2_score(y_hat_train, y_train)
print('\nThe model has the following metrics in train:')
print('MSE: ', mse_train)
print('RMSE: ', rmse_train)
print('MAE: ', mae_train)
print('R2: ', r2_train)

# Model in test
mse_test = mean_squared_error(y_hat_test, y_test)
mae_test = mean_absolute_error(y_hat_test, y_test)
r2_test = r2_score(y_hat_test, y_test)
rmse_test = np.sqrt(mse_test)
print('\nThe model has the following metrics in test:')
print('MSE: ', mse_test)
print('RMSE: ', rmse_test)
print('MAE: ', mae_test)
print('R2: ', r2_test)
print('')
