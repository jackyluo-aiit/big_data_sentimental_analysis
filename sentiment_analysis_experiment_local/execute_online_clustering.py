from sentiment_analysis_experiment_local.online_clustering import online_kmeans_pipeline
import pandas as pd
# pd.set_option('display.float_format', lambda x: '%.2f' % x)
import numpy as np
# np.set_printoptions(suppress=True)
import csv
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


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
                if item in original_data.index:
                    if key == 1 and original_data.loc[item, ['polarity']].values < 0:
                        count += 1
                    if key == 0 and original_data.loc[item, ['polarity']].values > 0:
                        count += 1
                else:
                    print("error:", item)
            cluster_fp = count / len(cluster_set)
            cluster_fp_set[key] = cluster_fp
    print("fpp:\n", cluster_fp_set)
    with open(result_filename, 'a+') as f:
        w = csv.writer(f)
        # w.writerows('using re-processed dataset:')
        w.writerows(cluster_fp_set.items())


data = pd.read_csv("database_data.csv", encoding="utf-8")
data.columns = ['database_id', '', '', '', 'content', '', 'polarity', '', '', '', '', '', '', '', '', '', '', 'label', '']
data = data.set_index('database_id')
# data = data.drop(index='root')
# data.index = data.index.astype('float')
data.index = data.index.astype('int64')
print(data.index)

okp = online_kmeans_pipeline()

i = 1
for index, row in data.iterrows():
    # if i > 20:
    #     break
    print('\nprocessing:', i)
    raw_content = pd.DataFrame(columns=['in_index', 'content'])
    raw_content = raw_content.append(pd.DataFrame({'in_index': index, 'content': [row['content']]}))
    print(raw_content)
    result_df = okp.process_content(raw_content)
    try:
        result_df.columns = ['label']
    except AttributeError:
        print("Discard this tweet!!!")
        continue
    # result_df = result_df.set_index('index')
    # label = result_df['label']
    # print(dist_base)
    # print(dist_ne)
    # print(dist_po)
    i += 1


okp.saveresult()

df = pd.read_csv('cluster_result_new_test.csv', header=None)
df.columns = ['index', 'label', 'sentimental_score']
df = df.set_index('index')
label_dict = {}

X = np.loadtxt("embed_mat.txt")
pca = PCA(n_components=2).fit(X)
datapoint = pca.transform(X)
plt.figure
label1 = ["r", "b"]
label = df['label']
color = [label1[i] for i in label]
plt.scatter(datapoint[:, 0], datapoint[:, 1], c=color)

centroids = np.loadtxt('mean_centroid.txt')
centroidpoint = pca.transform(centroids)
plt.scatter(centroidpoint[:, 0], centroidpoint[:, 1], marker='^', s=150, c='#000000')
# plt.show()

plt.savefig('cluster_result.png')

# ne_list=[]
# po_list=[]
# for index, row in df.iterrows():
#     if row['label'] == 0:
#         ne_list.append(index)
#     else:
#         po_list.append(index)