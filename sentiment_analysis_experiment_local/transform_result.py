import csv
import pandas as pd


df_result = pd.read_csv('cluster_result.csv')
df_result.columns = ['index', 'label']
df_result = df_result.set_index('index')

df_origin = pd.read_csv('database_data.csv')
df_origin.columns = ['database_id', '', '', '', 'content', '', 'polarity', '', '', '', '', '', '', '', '', '', '', 'label', '']
df_origin = df_origin.set_index('database_id')
df_origin.index.astype('int64')

print("clustering result info:")
print(df_result.info())
print("origin info:")
print(df_origin.info())

for index, label in df_result.iterrows():
    label = int(label)
    print(label)
    df_origin.loc[index, ['label']] = str(label)
    print("index: ", index,
          "label:", df_origin.loc[index, ['label']].values)

df_origin.to_csv("database_data_clustered.csv")
print('File saved!')