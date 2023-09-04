#!/usr/bin/env python
# coding: utf-8

# # Data Analysis Maxims Forced Choice (Flan-T5-xxl)

# In[27]:


import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[6]:


os.chdir("./results/fc")


# In[7]:


csv_files = [f for f in os.listdir() if f.endswith('.csv')]
print(csv_files)


# In[8]:


dfs = []

for csv in csv_files:
    df = pd.read_csv(csv)
    dfs.append(df)


# In[9]:


final_df = pd.concat(dfs, ignore_index=True)


# In[10]:


display(final_df)


# In[11]:


path = os.getcwd()
print(path)


# In[12]:


# Function to compare and populate the 'correct' column
def compare_and_fill(row):
    if row['phenomenon'] == 'Content':
        return row['true_answer'] == row['generation']
    elif row['phenomenon'] in ['Number', 'Both']:
        return row['true_answer_key'] == row['generation']
    else:
        return False

# Apply the function to each row and populate the 'correct' column
final_df['correct'] = final_df.apply(compare_and_fill, axis=1)

# Display the updated dataframe
#print(final_df)


# In[13]:


row_count = len(final_df.index)
print(row_count)


# In[14]:


#display(final_df)
final_df


# In[15]:


# Save the dataframe
#final_df.to_csv('./analysis/dataframe_fc.csv', index=False, mode='w')
final_df.to_csv('./analysis', index=False, mode='w')


# # Human Data

# In[16]:


def read_human_data():
    dfs = []
    df_dir = "./human_data/Human_Maxims.csv"
    df = pd.read_csv(df_dir)
    dfs.append(df)
    df = pd.concat(dfs).drop("Unnamed: 0", axis=1).reset_index().drop("index", axis=1)
    df = df.rename(columns={"itemNum": "item_id", "Correct": "correct", "Task": "phenomenon", "OptionChosen": "answer"})
    df.answer = df.answer.apply(lambda x: int(x[-1]))
    #df.phenomenon = df.phenomenon.map(PHENOMENA_PRETTY)
    return df


# In[17]:


human_df = read_human_data()
print("Number of human participants:", human_df.pKey.nunique())


# In[18]:


display(human_df)


# In[19]:


# Rename some column names
human_df.rename(
    columns={"item_id": "item_number", "answer": "generation"},
    inplace=True,
)

# Transform multiple integers to boolean
human_df = human_df.astype({'correct': bool})  

# Change phenomenon to "Human"
human_df.replace("MV", "Human", inplace=True)

display(human_df)


# # Concatinate Model and Human Data

# In[20]:


main_df = pd.concat([final_df, human_df])
#main_df.head()
# Save the dataframe
main_df.to_csv('./dataframe_main-fc.csv', index=False, mode='w')
main_df


# # Analyzing the dataframe and creating graphs

# # Correctness:

# In[21]:


# get the number of values

count_number = main_df['phenomenon'].value_counts()['Number']
count_content = main_df['phenomenon'].value_counts()['Content']
count_both = main_df['phenomenon'].value_counts()['Both']
count_human = main_df['phenomenon'].value_counts()['Human']

print(count_number, count_content, count_both, count_human)


# In[22]:


# count correct answers
print("True:", main_df['correct'].sum())
print("False:", (~main_df['correct'].sum())*-1)
#print(main_df.groupby('phenomenon').count())


# In[23]:


def cor_task(df):
    df = df.groupby("phenomenon")["correct"].apply(lambda x: (x == True).sum() / x.count())
    return df

cor_val = cor_task(main_df)


# In[24]:


print(cor_val.index)
#print(type(cor_val))


# In[25]:


# Get number of correct answers per phenomenon
phenomenon_dict = {}
for index, row in main_df.iterrows():
    if row["phenomenon"] not in phenomenon_dict:
        phenomenon_dict[row["phenomenon"]] = 0
    if row["correct"] == True:
        phenomenon_dict[row["phenomenon"]] += 1

print(phenomenon_dict)


# In[28]:


# Create a graph
cor_val.index=["Both", "Content", "Human", "Number"]
# Create a figure and axis
fig, ax = plt.subplots()

# Set the y-axis limits to range from 0 to 1
ax.set_ylim(0, 1)

# Create the bar plot
paired = sns.color_palette("Paired")
colors = [paired[1], paired[7], paired[2], paired[11]]  # Assign colors to the bars
bars = ax.bar(cor_val.index, cor_val.values, color=colors)

# Label the axes and give a title to the plot
plt.xlabel("Categories")
plt.ylabel("Proportion of Correct Answers")
plt.title("Correctness of Answers")

# Save the plot as PDF
plot_filename = './analysis/figures/correctness_fc.pdf'
plt.savefig(plot_filename, bbox_inches='tight')

# Show the plot
plt.show()




