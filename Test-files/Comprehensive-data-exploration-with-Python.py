# Introduction: In this exercise, I will create and submit predictions
# for the Housing Prices competition for Kaggle users.

# Import helpful libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats

# path to file I will use for predictions
test_data_path = 'train.csv'

# read test data file using pandas
test_data = pd.read_csv(test_data_path)

# check the decoration
print(test_data.columns)

# descriptive statistics summary
print(test_data['SalePrice'].describe())

# histogram
# sns.distplot(test_data['SalePrice'])
# plt.show()

# skewness and kurtosis
print("Skewness: %f" % test_data['SalePrice'].skew())
print("Kurtosis: %f" % test_data['SalePrice'].kurt())

# scatter plot grlivarea/saleprice
var = 'GrLivArea'
data = pd.concat([test_data['SalePrice'], test_data[var]], axis=1)
# data.plot.scatter(x=var, y='SalePrice', ylim=(0, 800000))
# plt.show()

# scatter plot totalbsmtsf/saleprice
var = 'TotalBsmtSF'
data = pd.concat([test_data['SalePrice'], test_data[var]], axis=1)
# data.plot.scatter(x=var, y='SalePrice', ylim=(0, 800000))
# plt.show()

# box plot overallqual/saleprice
var = 'OverallQual'
data = pd.concat([test_data['SalePrice'], test_data[var]], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000)
# plt.show()

var = 'YearBuilt'
data = pd.concat([test_data['SalePrice'], test_data[var]], axis=1)
f, ax = plt.subplots(figsize=(16, 8))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000)
# plt.xticks(rotation=90)
# plt.show()

# correlation matrix
corrmat = test_data.corr()
f, ax = plt.subplots(figsize=(12, 9))
# sns.heatmap(corrmat, vmax=.8, square=True)
# plt.show()

# saleprice correlation matrix
k = 10
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(test_data[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={
              'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
# plt.show()

# scatterplot
sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF',
        'FullBath', 'YearBuilt']
# sns.pairplot(test_data[cols], size=2.5)
# plt.show()

# missing data
total = test_data.isnull().sum().sort_values(ascending=False)
percent = (test_data.isnull().sum()/test_data.isnull().count()).sort_values(
                                                            ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)

# dealing with missing data
test_data = test_data.drop((missing_data[missing_data['Total'] > 1]).index, 1)
test_data = test_data.drop(
                        test_data.loc[test_data['Electrical'].isnull()].index)
# just checking that there's no missing data missing...
test_data.isnull().sum().max()

# standardizing data
saleprice_scaled = StandardScaler().fit_transform(
                                        test_data['SalePrice'][:, np.newaxis])
low_range = saleprice_scaled[saleprice_scaled[:, 0].argsort()][:10]
high_range = saleprice_scaled[saleprice_scaled[:, 0].argsort()][-10:]
print('outer range (low) of the distribution:')
print(low_range)
print('\nouter range (high) of the distribution:')
print(high_range)

# bivariate analysis saleprice/grlivarea
var = 'GrLivArea'
data = pd.concat([test_data['SalePrice'], test_data[var]], axis=1)
# data.plot.scatter(x=var, y='SalePrice', ylim=(0, 800000))
# plt.show()

# deleting points
test_data.sort_values(by='GrLivArea', ascending=False)[:2]
test_data = test_data.drop(test_data[test_data['Id'] == 1299].index)
test_data = test_data.drop(test_data[test_data['Id'] == 524].index)

# bivariate analysis saleprice/grlivarea
var = 'TotalBsmtSF'
data = pd.concat([test_data['SalePrice'], test_data[var]], axis=1)
# data.plot.scatter(x=var, y='SalePrice', ylim=(0, 800000))
# plt.show()

# histogram and normal probability plot
sns.distplot(test_data['SalePrice'], fit=norm)
fig = plt.figure()
res = stats.probplot(test_data['SalePrice'], plot=plt)
plt.show()

# applying log transformation
test_data['SalePrice'] = np.log(test_data['SalePrice'])

# transformed histogram and normal probability plot
sns.distplot(test_data['SalePrice'], fit=norm)
fig = plt.figure()
res = stats.probplot(test_data['SalePrice'], plot=plt)
plt.show()

# data transformation
test_data['GrLivArea'] = np.log(test_data['GrLivArea'])

# transformed histogram and normal probability plot
sns.distplot(test_data['GrLivArea'], fit=norm)
fig = plt.figure()
res = stats.probplot(test_data['GrLivArea'], plot=plt)
plt.show()

# histogram and normal probability plot
sns.distplot(test_data['TotalBsmtSF'], fit=norm)
fig = plt.figure()
res = stats.probplot(test_data['TotalBsmtSF'], plot=plt)
plt.show()

# create column for new variable
# if area>0 it gets 1, for area==0 it gets 0
test_data['HasBsmt'] = pd.Series(
                        len(test_data['TotalBsmtSF']), index=test_data.index)
test_data['HasBsmt'] = 0
test_data.loc[test_data['TotalBsmtSF'] > 0, 'HasBsmt'] = 1

# transform data
test_data.loc[test_data['HasBsmt'] == 1, 'TotalBsmtSF'] = np.log(
                                                    test_data['TotalBsmtSF'])
# histogram and normal probability plot
sns.distplot(test_data[test_data['TotalBsmtSF'] > 0]['TotalBsmtSF'], fit=norm)
fig = plt.figure()
res = stats.probplot(
            test_data[test_data['TotalBsmtSF'] > 0]['TotalBsmtSF'], plot=plt)
plt.show()

# scatter plot
plt.scatter(test_data['GrLivArea'], test_data['SalePrice'])
plt.show()

# scatter plot
plt.scatter(
            test_data[test_data['TotalBsmtSF'] > 0]['TotalBsmtSF'],
            test_data[test_data['TotalBsmtSF'] > 0]['SalePrice'])
plt.show()

# convert categorical variable into dummy
test_data = pd.get_dummies(test_data)
