from sentiment_analysis_experiment_local.online_clustering import online_kmeans_pipeline
import pandas as pd
import numpy as np
import csv


def computefpp(cluster_dict, original_data, result_filename, model=0):
    cluster_fp_set = {}
    for key in cluster_dict.keys():
        count = 0
        cluster_set = cluster_dict[key]
        if model == 0:
            for item in cluster_set:
                if key != 0:
                    key = 4
                if original_data.loc[item, ['label']].values != key:
                    count += 1
                cluster_fp = count / len(cluster_set)
                cluster_fp_set[key] = cluster_fp
        if model == 1:
            for item in cluster_set:
                if key == 1 and original_data.loc[item, ['polarity']].values <= 0:
                    count += 1
                if key == 0 and original_data.loc[item, ['polarity']].values > 0:
                    count += 1
            cluster_fp = count / len(cluster_set)

        cluster_fp_set[key] = cluster_fp
    print("fpp:\n", cluster_fp_set)
    with open(result_filename, 'a+') as f:
        w = csv.writer(f)
        # w.writerows('using re-processed dataset:')
        w.writerows(cluster_fp_set.items())


data = pd.read_csv("database_data.csv", encoding="utf-8")
data.columns = ['', '', '', '', 'content', '', 'polarity', '', '', '', '', '', '', '', '', '', '', 'label', '']
raw_content = pd.DataFrame(columns=['in_index', 'content'])
okp = online_kmeans_pipeline()
for index, row in data.iterrows():
    raw_content = pd.DataFrame(columns=['in_index', 'content'])
    raw_content = raw_content.append(pd.DataFrame({'in_index': [index], 'content': [row['content']]}))
    print(raw_content)
    okp.process_content(raw_content)
okp.saveresult()
df = pd.read_csv('cluster_result.csv')
print(df)
label_dict = {}

ne_list=[]
po_list=[]
for index, row in df.iterrows():
    if row['label'] == 0:
        ne_list.append(row['in_index'])
    else:
        po_list.append(row['in_index'])

label_dict[0] = ne_list
label_dict[1] = po_list
print(label_dict)
computefpp(label_dict, data, "mean-result_online_clustering.csv", model=1)

