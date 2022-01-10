import math
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


from scipy import stats
from scipy.stats import norm, skew

import pickle

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import explained_variance_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, \
                                 RidgeCV, Lasso, LassoCV, \
                                 ElasticNet, ElasticNetCV

df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

# look for train first 5 rows
df_train.head(5)

# look for test first 5 rows
df_test.head(5)

print(df_train.shape)
print(df_test.shape)

fig = plt.figure(figsize = (18,10))

fig.add_subplot(121)
plt.scatter(x = df_train['GrLivArea'], y = df_train['SalePrice'], color = "g", edgecolor = 'k')
plt.xlabel("GrLivArea")
plt.ylabel("SalePrice")

fig.add_subplot(122)
plt.scatter(x = df_train['TotalBsmtSF'], y = df_train['SalePrice'], color = "m", edgecolor = 'k')
plt.xlabel("TotalBsmtSF")
plt.ylabel("SalePrice")

plt.show()

stats = df_train['SalePrice'].describe()
stats
def plot_distribution(df):
    fig = plt.figure(figsize = (20,10))
    df['SalePrice'].plot.kde(color = 'r')
    df['SalePrice'].plot.hist(density = True, color = 'blue', edgecolor = 'k', bins = 100)
    plt.legend(['Normal distibution, ($\mu =${:.2f} and $\sigma =${:.2f})'.format(stats[1], stats[2])], loc='best')
    plt.title("Frequency distribution plot")
    plt.xlabel("SalePrice")
    # I don't like "1e6" number notation, so style will be 'plain'
    plt.ticklabel_format(style = 'plain', axis = 'y')
    plt.ticklabel_format(style = 'plain', axis = 'x')
    plt.show()

    plot_distribution(df_train)

    df_train["SalePrice"] = np.log1p(df_train["SalePrice"])
plot_distribution(df_train)

cor_matrix = df_train.corr()
cor_matrix.style.background_gradient(cmap='coolwarm')

cor_matrix2 = cor_matrix["SalePrice"]
cor_matrix2 = cor_matrix2.to_frame()
cor_matrix2.style.background_gradient(cmap='coolwarm')

# choosing some columns for plotting pairplots
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF']
pd.plotting.scatter_matrix(df_train[cols], alpha=0.2, figsize=(25, 25), color = 'cyan', edgecolor='k')
plt.show()

total = df_train.isnull().sum().sort_values(ascending=False)
percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'InPercents'])
missing_data.head(35).style.background_gradient(cmap='autumn')
