import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn import preprocessing
from sklearn.decomposition import PCA
from math import sqrt
from sklearn import svm
import time

df = pd.read_csv("F:/MSFE/semester 1/machine learning in Fin lab/HW5/hw5_treasury yield curve data.csv")

#Part 1: Exploratory Data Analysis

#scatterplot matrix
cols = ['SVENF01', 'SVENF02', 'SVENF03', 'SVENF04', 'SVENF05']
sns.pairplot(df[cols], size=2.5)
plt.tight_layout()
plt.title('ScatterPlot Matrix')
plt.savefig('ScatterPlot Matrix')
plt.show()

#Histogram
plt.hist(df.loc[:, 'SVENF01'], bins=30)
plt.xlabel('SVENF01')
plt.ylabel('Count')
plt.title('Histogram of SVENF01')
plt.show()

# Quantile‐Quantile Plot
stats.probplot(df.loc[:, 'SVENF02'], dist='norm', plot=plt)
plt.xlabel('SVENF02')
plt.title('Quantile‐quantile plot of SVENF02')
plt.show()

#Heat map
cols_hm = df.columns[1:31]
cm = np.corrcoef(df[cols_hm].values.T)
sns.set(font_scale=0.7)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 5}, yticklabels=cols_hm, xticklabels=cols_hm)
plt.title('Heat map')
plt.savefig('Heat Map')
plt.show()

#Split data into training and test sets
X = df.iloc[:, 1:-1].values
y = df['Adj_Close'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

#Standardize the features
sc = StandardScaler()
sc.fit(X_train)
X_train = sc.transform(X_train)
X_test = sc.transform(X_test)

#Part 2: Perform a PCA on the Treasury Yield dataset

model = PCA()
pca_features_train = model.fit_transform(X_train)
pca_features_test = model.transform(X_test)

# Create PCA instance: model
model = PCA(n_components=3)

# Transform the scaled samples: pca_features
X_train3=model.fit_transform(X_train)
X_test3=model.transform(X_test)

print(model.explained_variance_)
print(sum(model.explained_variance_))
print(model.explained_variance_ratio_)
print(sum(model.explained_variance_ratio_))

#Part 3: Linear regression v. SVM regressor - baseline

#Linear regression

#30 attributes
print('start_reg_all', time.process_time())
reg_all = LinearRegression()
reg_all.fit(X_train, y_train)
print('end_reg_all', time.process_time())

reg_y_train_pred = reg_all.predict(X_train)
reg_y_test_pred = reg_all.predict(X_test)

reg_rmse = sqrt(mean_squared_error(y_train, reg_y_train_pred)), \
           sqrt(mean_squared_error(y_test, reg_y_test_pred))
reg_r2 = r2_score(y_train, reg_y_train_pred), r2_score(y_test, reg_y_test_pred)
print('RMSE train: %.3f, test: %.3f' % reg_rmse)
print('R^2 train: %.3f, test: %.3f' % reg_r2)

#3 PCs

print('start_reg3_all', time.process_time())
reg3_all = LinearRegression()
reg3_all.fit(X_train3, y_train)
print('end_reg3_all', time.process_time())

reg3_y_train_pred = reg3_all.predict(X_train3)
reg3_y_test_pred = reg3_all.predict(X_test3)

reg3_rmse = sqrt(mean_squared_error(y_train, reg3_y_train_pred)), \
            sqrt(mean_squared_error(y_test, reg3_y_test_pred))
reg3_r2 = r2_score(y_train, reg3_y_train_pred), r2_score(y_test, reg3_y_test_pred)
print('RMSE train: %.3f, test: %.3f' % reg3_rmse)
print('R^2 train: %.3f, test: %.3f' % reg3_r2)


#SVM

#30 attributes
print('start_svm_all', time.process_time())
svm_all = svm.SVR(kernel='poly')
svm_all.fit(X_train, y_train)
print('end_svm_all', time.process_time())

svm_y_train_pred = svm_all.predict(X_train)
svm_y_test_pred = svm_all.predict(X_test)

svm_rmse = sqrt(mean_squared_error(y_train,svm_y_train_pred)), \
           sqrt(mean_squared_error(y_test,svm_y_test_pred))
svm_r2 = r2_score(y_train, svm_y_train_pred), r2_score(y_test, svm_y_test_pred)
print('RMSE train: %.3f, test: %.3f' % svm_rmse)
print('R^2 train: %.3f, test: %.3f' % svm_r2)

#3 PCs
print('start_svm3_all', time.process_time())
svm3_all = svm.SVR(kernel='poly')
svm3_all.fit(X_train3, y_train)
print('end_svm3_all',  time.process_time())

svm3_y_train_pred = svm3_all.predict(X_train3)
svm3_y_test_pred = svm3_all.predict(X_test3)

svm3_rmse = sqrt(mean_squared_error(y_train, svm3_y_train_pred)), \
            sqrt(mean_squared_error(y_test, svm3_y_test_pred))
svm3_r2 = r2_score(y_train, svm3_y_train_pred), r2_score(y_test, svm3_y_test_pred)
print('RMSE train: %.3f, test: %.3f' % svm3_rmse)
print('R^2 train: %.3f, test: %.3f' % svm3_r2)

print("My name is Yunxian Duan")
print("My NetID is: yunxian2")
print("I hereby certify that I have read the University policy on Academic Integrity "
      "and that I am not in violation.")
