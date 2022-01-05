# Introduction: In this exercise, I will create and submit predictions
# for the Housing Prices competition for Kaggle users.

# Import helpful libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats

# path to file I will use for predictions
test_data_path = 'test.csv'

# read test data file using pandas
test_data = pd.read_csv(test_data_path)

# check the decoration
print(test_data.columns)
