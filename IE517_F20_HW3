import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import seaborn as sns

High_Yield = pd.read_csv("F:/MSFE/semester 1/machine learning in Fin lab/HW3/HY_Universe_corporate bond.csv")

#Histogram
plt.hist(High_Yield.loc[:, 'Maturity Type'], bins=30)
plt.xlabel('Maturity Type')
plt.ylabel('Count')
plt.title('Histogram of Maturity Type')
plt.show()

# Quantile‐Quantile Plot
stats.probplot(High_Yield.loc[:, 'Client_Trade_Percentage'], dist='norm', plot=plt)
plt.title('Quantile‐quantile plot of Client Trade Percentage')
plt.show()

#Scatter
x_scatter = High_Yield.loc[:, 'volume_trades']
y_scatter = High_Yield.loc[:, 'LiquidityScore']
plt.scatter(x_scatter, y_scatter)
plt.xlabel('Volume Trades')
plt.ylabel('Liquidity Score')
plt.title('Scatter Plot')
plt.show()

#Heat map
#calculate correlations between real-valued attributes
Hm = High_Yield.iloc[:, 20:29]
corMat = DataFrame(Hm.corr())
#visualize correlations using heatmap
plt.pcolor(corMat)
plt.title('Heat map')
plt.show()

#Boxplot
Bp1 = High_Yield.iloc[:, 31:35].values
plt.boxplot(Bp1)
plt.xlabel('Attribute Index')
plt.ylabel('Quartile Ranges')
plt.title('Boxplot')
plt.show()

print("My name is Yunxian Duan")
print("My NetID is: yunxian2")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")




