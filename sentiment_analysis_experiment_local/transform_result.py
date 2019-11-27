import csv
import pandas as pd


df_result = pd.read_csv('cluster_result_new.csv')
df_result.columns = ['index', 'label', 'sentimental_score']
df_result = df_result.set_index('index')

df_origin = pd.read_csv('database_data.csv')
df_origin.columns = ['database_id', '', '', '', 'content', '', 'polarity', '', '', '', '', '', '', '', '', '', '', 'label', '']
df_origin = df_origin.set_index('database_id')
df_origin.index.astype('int64')

print("clustering result info:")
print(df_result.info())
print("origin info:")
print(df_origin.info())

for index, row in df_result.iterrows():
    label = int(row['label'])
    sentimental_score = row['sentimental_score']
    print(sentimental_score)
    print(label)
    df_origin.loc[index, ['label']] = str(sentimental_score)
    print("index: ", index,
          "label:", df_origin.loc[index, ['label']].values)

df_origin.to_csv("database_data_clustered_new.csv")
print('File saved!')