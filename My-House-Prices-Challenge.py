# Imports
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import (
                            LinearRegression, RidgeCV, LassoCV, ElasticNetCV)
from sklearn.metrics import mean_squared_error, make_scorer
from scipy.stats import skew
from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns


# Load Data
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# View Data
print(train.head())
