import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn import preprocessing
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

df = pd.read_csv("F:/MSFE/semester 1/machine learning in Fin lab/HW4/housing.csv")

#EDA

#scatterplot matrix
cols = ['NOX', 'INDUS', 'RM', 'LSTAT', 'MEDV']
sns.pairplot(df[cols], size=2.5)
plt.tight_layout()
plt.title('ScatterPlot Matrix')
plt.savefig('ScatterPlot Matrix')
plt.show()

#Histogram
plt.hist(df.loc[:, 'RAD'], bins=30)
plt.xlabel('Index of accessibility to radical highways')
plt.ylabel('Count')
plt.title('Histogram of index of accessibility to radical highways')
plt.show()

# Quantile‐Quantile Plot
stats.probplot(df.loc[:, 'MEDV'], dist='norm', plot=plt)
plt.xlabel('MEDV')
plt.title('Quantile‐quantile plot of MEDV')
plt.show()

#Heat map
cols_hm = ['CRIM',	'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX',	'PTRATIO', 'B',	'LSTAT', 'MEDV']
cm = np.corrcoef(df[cols_hm].values.T)
sns.set(font_scale=1)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 8}, yticklabels=cols_hm, xticklabels=cols_hm)
plt.title('Heat map')
plt.savefig('Heat Map')
plt.show()


#Split data into training and test sets
X = df.iloc[:, :-1].values
y = df['MEDV'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
sc = preprocessing.StandardScaler().fit(X_train)
X_train = sc.transform(X_train)
X_test = sc.transform(X_test)

#Linear regression
reg_all = LinearRegression()
reg_all.fit(X_train, y_train)
reg_coef = reg_all.coef_
print('Slope:', reg_coef)
reg_inter = reg_all.intercept_
print('Intercept:', reg_inter)

y_train_pred = reg_all.predict(X_train)
y_test_pred = reg_all.predict(X_test)

plt.scatter(y_train_pred, y_train_pred-y_train, c='steelblue', marker='o', edgecolor='white', label='Training data')
plt.scatter(y_test_pred, y_test_pred-y_test, c='limegreen', marker='s', edgecolor='white', label='Test data')
plt.xlabel('Predicted MEDV')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=50, colors='black', lw=2)
plt.xlim([-10, 50])
plt.show()

reg_mse = mean_squared_error(y_train, y_train_pred), mean_squared_error(y_test, y_test_pred)
reg_r2 = r2_score(y_train, y_train_pred), r2_score(y_test, y_test_pred)
print('MSE train: %.3f, test: %.3f' % reg_mse)
print('R^2 train: %.3f, test: %.3f' % reg_r2)

#Ridge regression
ridge_alpha_space = np.arange(0.01, 1, 0.05)
ridge_coef = []
ridge_inter = []
ridge_r2_train = []
ridge_r2_test = []
ridge_mse_train = []
ridge_mse_test = []

ridge = Ridge(normalize=True)

for alpha in ridge_alpha_space:
    ridge.alpha = alpha
    ridge.fit(X_train, y_train)
    ridge_coef.append(ridge.coef_)
    ridge_inter.append(ridge.intercept_)
    ridge_r2_train.append(r2_score(y_train, y_train_pred))
    ridge_r2_test.append(r2_score(y_test, y_test_pred))
    ridge_mse_train.append(mean_squared_error(y_train, y_train_pred))
    ridge_mse_test.append(mean_squared_error(y_test, y_test_pred))

ridge_best_mse = min(ridge_mse_test)
ridge_best_alpha = ridge_alpha_space[ridge_mse_test.index(ridge_best_mse)]
print('The best alpha:', ridge_best_alpha, 'Coefficient:', ridge_coef[ridge_mse_test.index(ridge_best_mse)],
      'Intercept:', ridge_inter[ridge_mse_test.index(ridge_best_mse)])

ridge_best = Ridge(alpha=ridge_best_alpha)
ridge_best.fit(X_train, y_train)
ridge_y_train_pred = ridge_best.predict(X_train)
ridge_y_test_pred = ridge_best.predict(X_test)

plt.scatter(ridge_y_train_pred, ridge_y_train_pred-y_train, c='steelblue', marker='o', edgecolor='white',
            label='Training data')
plt.scatter(ridge_y_test_pred, ridge_y_test_pred-y_test, c='limegreen', marker='s', edgecolor='white',
            label='Test data')
plt.xlabel('Predicted MEDV')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=50, colors='black', lw=2)
plt.xlim([-10, 50])
plt.show()

ridge_reg_mse = mean_squared_error(y_train, ridge_y_train_pred), mean_squared_error(y_test, ridge_y_test_pred)
ridge_reg_r2 = r2_score(y_train, ridge_y_train_pred), r2_score(y_test, ridge_y_test_pred)
print('MSE train: %.3f, test: %.3f' % ridge_reg_mse)
print('R^2 train: %.3f, test: %.3f' % ridge_reg_r2)

#LASSO
lasso_alpha_space = np.arange(0.01, 1, 0.05)
lasso_coef = []
lasso_inter = []
lasso_r2_train = []
lasso_r2_test = []
lasso_mse_train = []
lasso_mse_test = []

lasso = Lasso()

for alpha in lasso_alpha_space:
    lasso.alpha = alpha
    lasso.fit(X_train, y_train)
    lasso_coef.append(lasso.coef_)
    lasso_inter.append(lasso.intercept_)
    lasso_r2_train.append(r2_score(y_train, y_train_pred))
    lasso_r2_test.append(r2_score(y_test, y_test_pred))
    lasso_mse_train.append(mean_squared_error(y_train, y_train_pred))
    lasso_mse_test.append(mean_squared_error(y_test, y_test_pred))

lasso_best_mse = min(lasso_mse_test)
lasso_best_alpha = lasso_alpha_space[lasso_mse_test.index(lasso_best_mse)]
print('The best alpha:', lasso_best_alpha, 'Coefficient:', lasso_coef[lasso_mse_test.index(lasso_best_mse)],
      'Intercept:', lasso_inter[lasso_mse_test.index(lasso_best_mse)])

lasso_best = Lasso(alpha=lasso_best_alpha)
lasso_best.fit(X_train, y_train)
lasso_y_train_pred = lasso_best.predict(X_train)
lasso_y_test_pred = lasso_best.predict(X_test)

plt.scatter(lasso_y_train_pred, lasso_y_train_pred-y_train, c='steelblue', marker='o', edgecolor='white',
            label='Training data')
plt.scatter(lasso_y_test_pred, lasso_y_test_pred-y_test, c='limegreen', marker='s', edgecolor='white',
            label='Test data')
plt.xlabel('Predicted MEDV')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=50, colors='black', lw=2)
plt.xlim([-10, 50])
plt.show()

lasso_reg_mse = mean_squared_error(y_train, lasso_y_train_pred), mean_squared_error(y_test, lasso_y_test_pred)
lasso_reg_r2 = r2_score(y_train, lasso_y_train_pred), r2_score(y_test, lasso_y_test_pred)
print('MSE train: %.3f, test: %.3f' % lasso_reg_mse)
print('R^2 train: %.3f, test: %.3f' % lasso_reg_r2)


print("My name is Yunxian Duan")
print("My NetID is: yunxian2")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")
