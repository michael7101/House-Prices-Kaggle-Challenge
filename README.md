### House-Prices-Kaggle-Challenge

##### About

In this competition we use creative feature engineering and advanced regression techniques to predict house sales prices. The full details of the competition can be found at https://www.kaggle.com/c/house-prices-advanced-regression-techniques.

##### Practice Skills

Creative feature engineering
Advanced regression techniques like random forest and gradient boosting

##### Goal

To win the competition, I must build a model that predicts the SalePrice variables' value as accurately as possible for each Id in the test set.

##### Metric

Submissions are evaluated on Root-Mean-Squared-Error (RMSE) between the logarithm of the predicted value and the logarithm of the observed sales price. (Taking logs means that errors in predicting expensive houses and cheap houses will affect the result equally.)

### Imports
```
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
```

```
### View the traing data
train.head()
```

```
# Describe the traing data
train.describe()
```

###Read the data
X_full = pd.read_csv('train.csv', index_col='Id')
X_test_full = pd.read_csv('test.csv', index_col='Id')

###Remove rows with missing target, separate target from predictors
X_full.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = X_full.SalePrice
X_full.drop(['SalePrice'], axis=1, inplace=True)

###Break off validation set from training data
X_train_full, X_valid_full, y_train, y_valid = train_test_split(
    X_full, y, train_size=0.8, test_size=0.2, random_state=0)

###"Cardinality" means the number of unique values in a column
###Select categorical columns with relatively low cardinality
categorical_cols = [cname for cname in X_train_full.columns if
                    X_train_full[cname].nunique() < 10 and
                    X_train_full[cname].dtype == "object"]

###Select numerical columns
numerical_cols = [
    cname for cname in X_train_full.columns if X_train_full[cname].dtype in [
        'int64', 'float64']]

###Keep selected columns only
my_cols = categorical_cols + numerical_cols
X_train = X_train_full[my_cols].copy()
X_valid = X_valid_full[my_cols].copy()
X_test = X_test_full[my_cols].copy()

###Preprocessing for numerical data
numerical_transformer = SimpleImputer(strategy='constant')

###Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

###Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)])

###Define model
model = RandomForestRegressor(n_estimators=100, random_state=0)

###Bundle preprocessing and modeling code in a pipeline
clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('model', model)])

###Preprocessing of training data, fit model
clf.fit(X_train, y_train)

###Preprocessing of validation data, get predictions
preds = clf.predict(X_valid)

print('MAE:', mean_absolute_error(y_valid, preds))

###Preprocessing for numerical data
numerical_transformer = SimpleImputer(strategy='constant')

###Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

###Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)])

###Define model
model = RandomForestRegressor(n_estimators=100, random_state=0)

###Bundle preprocessing and modeling code in a pipeline
my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', model)])

###Preprocessing of training data, fit model
my_pipeline.fit(X_train, y_train)

###Preprocessing of validation data, get predictions
preds = my_pipeline.predict(X_valid)

###Evaluate the model
score = mean_absolute_error(y_valid, preds)
print('MAE:', score)

###Preprocessing of test data, fit model
preds_test = my_pipeline.predict(X_test)

###Save test predictions to file
output = pd.DataFrame({'Id': X_test.index,
                       'SalePrice': preds_test})
output.to_csv('submission3.csv', index=False)
