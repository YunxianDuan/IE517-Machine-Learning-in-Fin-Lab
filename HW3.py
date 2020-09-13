import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import seaborn as sns

High_Yield = pd.read_csv("F:/MSFE/semester 1/machine learning in Fin lab/HW3/HY_Universe_corporate bond.csv")

#Histogram
plt.hist(High_Yield.loc[:, 'Client_Trade_Percentage'], bins=30)
plt.xlabel('Client Trade Percentage')
plt.ylabel('Count')
plt.title('Histogram of Client Trade Percentage')
plt.show()

# Quantile‐Quantile Plot
stats.probplot(High_Yield.loc[:, 'weekly_median_volume'], dist="norm", plot=plt)
plt.title('Quantile‐quantile plot of Weekly Median Volume')
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
sns.boxplot(x='bond_type', y='Client_Trade_Percentage', data=High_Yield)
plt.xlabel('Bond Type')
plt.ylabel('Client Trade Percentage')
plt.title('Boxplot')
plt.show()

print("My name is Yunxian Duan")
print("My NetID is: yunxian2")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")




