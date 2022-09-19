# Random Forest implementation -> Use of RandomForestRegressor from Sklearn

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
* In this case, all variables ares used to make a random forest model.

With the parameters used, the metrics used for the model are MSE and RMSE: 
    - MSE: 40.96
    - RMSE: 6.39

At the end, the graphs of the model are shown. 
'''

# Used libraries
import pandas               as pd
import numpy                as np 
import matplotlib.pyplot    as plt
import seaborn              as sns

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
#from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split #, KFold, cross_val_score, cross_val_predict, cross_validate, GridSearchCV
#from sklearn import metrics, feature_selection, model_selection
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
#from sklearn.tree import plot_tree


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
forest = RandomForestRegressor(n_estimators=15, max_depth=9, random_state=42)
forest.fit(X_train, y_train)

# Predictions
y_hat_train = forest.predict(X_train)
y_hat_test = forest.predict(X_test)

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
print("\nThe model has the following metrics in train:")
print("MSE: ", mse_train)
print("RMSE: ", rmse_train)
print("MAE: ", mae_train)
print("R2: ", r2_train)

# Model in test
mse_test = mean_squared_error(y_hat_test, y_test)
mae_test = mean_absolute_error(y_hat_test, y_test)
r2_test = r2_score(y_hat_test, y_test)
rmse_test = np.sqrt(mse_test)
print("\nThe model has the following metrics in train:")
print("MSE: ", mse_test)
print("RMSE: ", rmse_test)
print("MAE: ", mae_test)
print("R2: ", r2_test)
print('')

# Graphing the model
'''plt.figure(figsize=(20, 5))
plot_tree(forest, max_depth=2, feature_names=X.columns)
plt.show()

plt.barh(y=X.columns, width=forest.feature_importances_,)
plt.show()'''