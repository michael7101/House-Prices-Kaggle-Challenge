<<<<<<< Updated upstream
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
sns.distplot(test_data['SalePrice'])
plt.show()

# skewness and kurtosis
print("Skewness: %f" % test_data['SalePrice'].skew())
print("Kurtosis: %f" % test_data['SalePrice'].kurt())

# scatter plot grlivarea/saleprice
var = 'GrLivArea'
data = pd.concat([test_data['SalePrice'], test_data[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0, 800000))
plt.show()

# scatter plot totalbsmtsf/saleprice
var = 'TotalBsmtSF'
data = pd.concat([test_data['SalePrice'], test_data[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0, 800000))
plt.show()

# box plot overallqual/saleprice
var = 'OverallQual'
data = pd.concat([test_data['SalePrice'], test_data[var]], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000)
plt.show()

var = 'YearBuilt'
data = pd.concat([test_data['SalePrice'], test_data[var]], axis=1)
f, ax = plt.subplots(figsize=(16, 8))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000)
plt.xticks(rotation=90)
plt.show()

# correlation matrix
corrmat = test_data.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True)
plt.show()

# saleprice correlation matrix
k = 10
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(test_data[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={
                 'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()
=======
# Import helpful libraries
import pandas as pd
import math
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# Load the data, and separate the target
iowa_file_path = 'train.csv'
home_data = pd.read_csv(iowa_file_path)
y = home_data.SalePrice

# Create X (After completing the exercise, you can return to modify this line!)
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath']

# Select columns corresponding to features, and preview the data
X = home_data[features]
X.head()

# Split into validation and training data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# Define a random forest model
rf_model = RandomForestRegressor(random_state=1)
rf_model.fit(train_X, train_y)
rf_val_predictions = rf_model.predict(val_X)
rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)

print("Validation MAE for Random Forest Model: {:,.0f}".format(rf_val_mae))

# test pandas
#file = pd.read_csv('train.csv')
#print(file.head(20))
#pti
>>>>>>> Stashed changes
