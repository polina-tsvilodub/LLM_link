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


# In[36]:


# Initialize an empty list to store DataFrames
dfs = []

# Read and append the CSV files to the list of DataFrames
for csv_file in csv_files:
    df = pd.read_csv(csv_file, delimiter=";")
    dfs.append(df)


# In[50]:


final_df = pd.concat(dfs, ignore_index=True)
#display(final_df)


# In[51]:


# Transform integers to boolean
final_df = final_df.astype({'annotation': bool})
display(final_df)


# Human Data

# In[58]:


def read_human_data():
    dfs = []
    df_dir = "C:/Users/shagr/OneDrive/Desktop/LLM-Project/github/LLM_link-master/human_data/Human_Maxims.csv"
    df = pd.read_csv(df_dir)
    dfs.append(df)
    df = pd.concat(dfs).drop("Unnamed: 0", axis=1).reset_index().drop("index", axis=1)
    df = df.rename(columns={"itemNum": "item_id", "Correct": "correct", "Task": "phenomenon", "OptionChosen": "answer"})
    df.answer = df.answer.apply(lambda x: int(x[-1]))
    #df.phenomenon = df.phenomenon.map(PHENOMENA_PRETTY)
    return df

human_df = read_human_data()
#display(human_df)


# In[59]:


# Rename some column names
human_df.rename(
    columns={"item_id": "item_number", "correct": "annotation", "answer": "generation"},
    inplace=True,
)

# Transform multiple integers to boolean
human_df = human_df.astype({'annotation': bool})  

# Change phenomenon to "Human"
human_df.replace("MV", "Human", inplace=True)

display(human_df)


# In[60]:


main_df = pd.concat([final_df, human_df])
#main_df.head()
# Save the dataframe
#main_df.to_csv('./dataframe_main-fc.csv', index=False, mode='w')
main_df


# # Create dataframe

# In[67]:


# Filter rows with NaN in the 'phenomenon' column and calculate the proportion of True values
model_nan_df = main_df[main_df['phenomenon'].isna()]
model_nan_true_count = model_nan_df['annotation'].sum()
model_nan_total_count = len(model_nan_df)
model_nan_proportion = model_nan_true_count / model_nan_total_count

# Filter rows with 'Human' in the 'phenomenon' column and calculate the proportion of True values
human_df = main_df[main_df['phenomenon'] == 'Human']
human_true_count = human_df['annotation'].sum()
human_total_count = len(human_df)
human_proportion = human_true_count / human_total_count

# Create a bar plot
labels = ['Model', 'Human']
proportions = [model_nan_proportion, human_proportion]

plt.figure(figsize=(8, 6))
paired = sns.color_palette("Paired")
#colors = [paired[2], paired[11]]
plt.bar(labels, proportions, color=[paired[11], paired[2]])
plt.title('Correct Answer Proportions')
plt.ylabel('Proportion')

# Save the plot as a file 
plot_filename = 'C:/Users/shagr/OneDrive/Desktop/LLM-Project/github/LLM_link-master/analysis/figures/answersprop_free.pdf'
plt.savefig(plot_filename, bbox_inches='tight')

# Display the plot
plt.show()
print("Model: " , model_nan_proportion , ", Human: " , human_proportion)


# In[68]:


# # Calculate the proportions of True and False values
# correct_count = final_df['annotation'].sum()
# total_count = len(final_df)
# incorrect_count = total_count - correct_count

# correct_proportion = correct_count / total_count
# incorrect_proportion = incorrect_count / total_count

# # Create a bar plot
# labels = ['Correct', 'Incorrect']
# proportions = [correct_proportion, incorrect_proportion]
# paired = sns.color_palette("Paired")
# colors = [paired[2], paired[11]]

# plt.figure(figsize=(8, 6))
# plt.bar(labels, proportions, color=colors)
# plt.title('Correctness Free Production')
# plt.xlabel('Correctness')
# plt.ylabel('Proportion')
# plt.ylim(0, 1)  # Set the y-axis scale to 0-1

# # Save the plot as a file 
# plot_filename = 'C:/Users/shagr/OneDrive/Desktop/LLM-Project/github/LLM_link-master/analysis/figures/answers_free.pdf'
# plt.savefig(plot_filename, bbox_inches='tight')

# # Display the plot
# plt.show()
# print("Correct: " , correct_proportion , ", Incorrect: " , incorrect_proportion)


# In[ ]:




