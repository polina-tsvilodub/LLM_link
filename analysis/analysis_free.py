#!/usr/bin/env python
# coding: utf-8

# # Data Analysis Maxims Free Production (Flan-T5 XXL)

# In[29]:


import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob


# In[34]:


# Specify the path to the folder containing your CSV files
folder_path = 'C:/Users/shagr/OneDrive/Desktop/LLM-Project/github/LLM_link-master/results/free'

# Use glob to find CSV files ending with "...annotated.csv"
csv_files = glob.glob(os.path.join(folder_path, '*annotated.csv'))

print(csv_files)


# In[35]:


# Initialize an empty list to store DataFrames
dfs = []

# Read and append the CSV files to the list of DataFrames
for csv_file in csv_files:
    df = pd.read_csv(csv_file)
    dfs.append(df)


# In[32]:


final_df = pd.concat(dfs, ignore_index=True)
display(final_df)


# In[ ]:




