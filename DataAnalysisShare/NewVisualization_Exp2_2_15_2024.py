# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 14:39:57 2024

@author: hadar

This code will create a box plot for each metric, with each box representing 
    a condition (baseline, stress, recovery). 
The mean value for each condition is marked on the plot, with text labels above the mean line. 
The plots are faceted by metric and grouped into one figure. 
Each condition is represented by a different color.

1. Pre-plot: subset the data to the selected metrics and change column names, and melt the dataframe to a long format.
2. Creates a box plot for each metric (6 metrics), with each bar representing a condition (3 conditions: baseline, stress, recovery.
3. Mark the mean value and add text labels for the mean value above the mean line (in a way it will not overlap with the shapes)
4. Facets the plot by metric.
5. Create a title for each plot with the name of the metric (not same as the variable name)
6. Group the plot into 1 figure.
"""

import seaborn as sns
import matplotlib.pyplot as plt

# Sample DataFrame
import pandas as pd
import numpy as np

#%%
# get data
general_path = 'C:/users/hadar/OneDrive/Documents/3_MASTER/2-RESEARCH/GarminValidation_Exp2'
data_path = f'{general_path}/Data_Analysis'
res_path = f'{data_path}/Results'

df_all = pd.read_csv(f'{res_path}/df_all.csv')
df_all.shape
print(df_all.sample(10))

df_mean_melt = pd.read_csv(f'{res_path}/df_mean_melt.csv')

selected_metrics = ['Mean_HR_bpm','RMSSD_ms','SD2_SD1_ratio','LF_HF_ratio_AR', 'HFpow_AR_nu', 'garmin_stress']                 

#%%
#################    Pre-plot


# The names that will be presented on the plots
plotnames = ['Garmin Stress Score', 'Mean HR', 'SD2/SD1', 'LF/HF', 'RMSSD', 'HF power[nu]']                   

# replace column names 
data = df_all.rename(columns={'sub': 'subject', "part": "condition"})
# replace the 'part' variables to the condition names
data['condition'] = data['condition'].replace({'p1': 'Baseline', 'p2': 'Stress', 'p3': 'Recovery'})
# reshape dataframe (melt)
data_long = pd.melt(data, id_vars=['subject', 'condition'], value_vars=selected_metrics, var_name='metric', value_name='value')

###### if using the means data (single value for each subject in each condition)
df = pd.read_csv(f'{res_path}/data_for_anova.csv')
data_means_long = pd.melt(df, id_vars=['subject', 'condition'], value_vars=selected_metrics, var_name='metric', value_name='value')

#%%
#################   Plot



"""
# Creating sample data
np.random.seed(0)
data = {
    'subject': np.random.choice(['A', 'B', 'C'], size=100),
    'condition': np.random.choice(['baseline', 'stress', 'recovery'], size=100),
    'metric': np.random.choice(['garmin_stress','Mean_HR_bpm','SD2_SD1_ratio','LF_HF_ratio_AR', 'RMSSD_ms','HFpow_AR_nu'], size=100),
    'value': np.random.normal(size=100)
}
df = pd.DataFrame(data)
"""
# Mapping metric names
metric_names = {
    'garmin_stress': 'Garmin Stress Score',
    'Mean_HR_bpm': 'Mean HR',
    'SD2_SD1_ratio': 'SD2/SD1',
    'LF_HF_ratio_AR': 'LF/HF',
    'RMSSD_ms': 'RMSSD',
    'HFpow_AR_nu': 'HF power[nu]'
}

#%%

# Plotting
def fun_create_boxplots(df):

    g = sns.FacetGrid(df, col='metric', col_wrap=3, sharey=False, height=4)
    g.map_dataframe(sns.boxplot, x='condition', y='value', palette='Set2')
    
    # Adding mean value
    for ax, metric in zip(g.axes, df['metric'].unique()):
        ax.set_title(metric_names[metric])
        for i, condition in enumerate(df['condition'].unique()):
            mean_val = df[(df['metric'] == metric) & (df['condition'] == condition)]['value'].mean()
            ax.text(i, mean_val, f'{mean_val:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
#%%

# fun_create_boxplots(data_long)
fun_create_boxplots(data_means_long)

#%%


#%%

########  plot boxplots for the means data - clean outliers (distant more than 3 sd of mean)

import pandas as pd
import numpy as np

df = df_mean_melt[df_mean_melt['variable'].isin(selected_metrics)]
df = df.rename(columns={'sub': 'subject', "part": "condition", "variable": "metric"})
# replace the 'part' variables to the condition names
df['condition'] = df['condition'].replace({'p1': 'Baseline', 'p2': 'Stress', 'p3': 'Recovery'})

# Define threshold (e.g., 3 standard deviations from the mean)
threshold = 3

# Function to remove outliers for a specific condition and metric
def remove_outliers(df):
    cleaned_dfs = []
    for condition, metric in df.groupby(['condition', 'metric']):
        condition_mask = df['condition'] == condition[0]
        metric_mask = df['metric'] == condition[1]
        subset_df = df[condition_mask & metric_mask]
        subset_df['z_score'] = (subset_df['value'] - subset_df['value'].mean()) / subset_df['value'].std()
        cleaned_subset_df = subset_df[abs(subset_df['z_score']) <= threshold]
        cleaned_dfs.append(cleaned_subset_df.drop(columns=['z_score']))
    return pd.concat(cleaned_dfs)

# Remove outliers for all conditions and metrics
cleaned_df = remove_outliers(df)

order_con = ['Baseline', 'Stress', 'Recovery']
cleaned_df['condition'] = pd.Categorical(cleaned_df['condition'], categories=order_con, ordered=True)
order_metric = ['garmin_stress','Mean_HR_bpm','SD2_SD1_ratio','LF_HF_ratio_AR', 'RMSSD_ms','HFpow_AR_nu']
cleaned_df['metric'] = pd.Categorical(cleaned_df['metric'], categories=order_metric, ordered=True)
# Sort the DataFrame based on the new order
cleaned_df = cleaned_df.sort_values(by='condition')
cleaned_df = cleaned_df.sort_values(by='metric')


print(cleaned_df)

#%%

# Plotting
fun_create_boxplots(cleaned_df)



#%%






# **************             plot for pilot data   ********************



data_path = 'C:/users/hadar/OneDrive/Documents/3_MASTER/2-RESEARCH/experiment_validation'
res_path = f'{data_path}/Results'
df_mean_wide_edited_pilot = pd.read_csv(f'{res_path}/df_mean_wide_edited_pilot.csv')

#%%
#################    Pre-plot


# The names that will be presented on the plots
# plotnames = ['Garmin Stress Score', 'Mean HR', 'SD2/SD1', 'LF/HF', 'RMSSD', 'HF power[nu]']                   

# reshape dataframe (melt)

data_means_long = pd.melt(df_mean_wide_edited_pilot, id_vars=['subject', 'condition'], value_vars=selected_metrics, var_name='metric', value_name='value')

# Mapping metric names
metric_names = {
    'garmin_stress': 'Garmin Stress Score',
    'Mean_HR_bpm': 'Mean HR',
    'SD2_SD1_ratio': 'SD2/SD1',
    'LF_HF_ratio_AR': 'LF/HF',
    'RMSSD_ms': 'RMSSD',
    'HFpow_AR_nu': 'HF power[nu]'
}

#%%
fun_create_boxplots(data_means_long)
#%%


########  create a boxplot for the reported stress

df = pd.read_csv(f'{res_path}/df_mean_wide_edit.csv')



# Create a figure and axis object
fig, ax = plt.subplots(figsize=(4, 3))

# Draw boxplots for each condition
sns.boxplot(x='condition', y='reported_stress', data=df, ax=ax, palette='Set2')

# Set the title of the plot
ax.set_title("Verbal Report of Stress Levels after Each Task", pad=20)
ax.set_ylabel('Reported Stress')
# Show the plot
plt.show()


