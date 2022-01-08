# Set up Enivorment
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import skew
from scipy.stats.stats import pearsonr

# Load Data
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# View Data
print(train.head())

# Merging Data
all_data = pd.concat((train.loc[:, 'MSSubClass':'SaleCondition'],
                      test.loc[:, 'MSSubClass':'SaleCondition']))

# Data preprocessing
matplotlib.rcParams['figure.figsize'] = (12.0, 6.0)
prices = pd.DataFrame({
    "price": train["SalePrice"], "log(price + 1)": np.log1p(train[
                                                            "SalePrice"])})
prices.hist()
plt.show()
