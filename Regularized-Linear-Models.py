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
