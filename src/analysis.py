import pandas as pd
import sys, os
import matplotlib.pyplot as plt

'''
Python file to run data analysis and generate plots/tables from analysis on performance of model weights
'''

data_dir = "data/"
data_files = os.listdir(data_dir)

data = \
    [(*file.split('&'), pd.read_csv(data_dir + file)) for file in data_files]

## Boxplots for damage on Cowpunch Ez
box_data = []
labels = []
for env, model, weights, df in data:
    if env == "MineRLPunchCow-v0":
        labels.append(weights)
        box_data.append(df['damage'].values[0])
        
plt.boxplot(box_data, labels=labels)
plt.show()
