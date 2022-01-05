# set up

import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error
from sklearn.decomposition import PCA
from matplotlib import rcParams
from sklearn.preprocessing import QuantileTransformer


train = pd.read_csv("train.csv")
X_test = pd.read_csv("test.csv")
submission_data = pd.read_csv('sample_submission.csv')

house_df = pd.concat([train,X_test],ignore_index = True, sort = False)
tr_idx = house_df['SalePrice'].notnull()

# Standardize
# Create principal components
# Convert to dataframe
# Create loadings
# transpose the matrix of loadings
# so the columns are the principal components
# and the rows are the original features
def apply_pca(X, standardize=True):
    if standardize:
        X = (X - X.mean(axis=0)) / X.std(axis=0)
    pca = PCA()
    X_pca = pca.fit_transform(X)
    component_names = [f"PC{i+1}" for i in range(X_pca.shape[1])]
    X_pca = pd.DataFrame(X_pca, columns=component_names)
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=component_names,
        index=X.columns,)
    return pca, X_pca, loadings

    def outlier_iqr(data):
    q1,q3 = np.percentile(data,[25,75])
    iqr = q3-q1
    lower = q1-(iqr*2)
    upper = q3+(iqr*2)
    return np.where((data>upper)|(data<lower))

    house_df.shape

    house_df.info()

    house_df.head(3).style.set_properties(**{'background-color': 'Grey',
                           'color': 'white',
                           'border-color': 'darkblack'})

house_df.drop('Id',axis=1,inplace=True,errors='ignore')

house_df.describe().T.style.set_properties(**{'background-color': 'Grey',
                           'color': 'white',
                           'border-color': 'darkblack'})

plt.figure(figsize = (8,6))
ax = house_df.dtypes.value_counts().plot(kind='bar',grid = False,fontsize=20,color='grey')
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+ p.get_width() / 2., height + 1, height, ha = 'center', size = 25)
sns.despine()

# "Cardinality" meancategorical_colss the number of unique values in a column
# Select categorical columns with relatively low cardinality (convenient but arbitrary)
categorical_cols = [cname for cname in house_df.loc[:,:'SaleCondition'].columns if
                    house_df[cname].nunique() < 10 and
                    house_df[cname].dtype == "object"]

# Select numerical columns
int_cols = [cname for cname in house_df.loc[:,:'SaleCondition'].columns if
                house_df[cname].dtype in ['int64']]
float_cols = [cname for cname in house_df.loc[:,:'SaleCondition'].columns if
                house_df[cname].dtype in ['float64']]

numerical_cols = int_cols + float_cols

# Keep selected columns only
my_cols = categorical_cols + numerical_cols

import missingno as msno
msno.matrix(house_df[tr_idx])

from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=5)
house_df.loc[:,numerical_cols] = imputer.fit_transform(house_df.loc[:,numerical_cols])
msno.bar(house_df.loc[:,categorical_cols])
