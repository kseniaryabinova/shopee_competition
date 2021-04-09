import glob
import re

import cudf
import cupy
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from cuml.feature_extraction.text import TfidfVectorizer
from cuml.neighbors import NearestNeighbors

from sklearn.metrics import f1_score

print('Computing text embeddings...')


def get_embeddings_metric(df):
    for i in range(len(df)):
        df.loc[df.loc[i, 'preds'], 'pred_label'] = df.loc[i, 'label_group']
    f1 = f1_score(df['label_group'], df['pred_label'], average='macro')
    return f1


cpu_df = pd.read_csv('image/reliable_validation_tm.csv')
cpu_df = cpu_df[cpu_df['fold_strat'].isna()]

tmp = cpu_df.groupby('label_group').posting_id.agg('unique').to_dict()
cpu_df['target'] = cpu_df.label_group.map(tmp)

df = cudf.DataFrame(cpu_df)


# model = TfidfVectorizer(stop_words=None,
#                         binary=True,
#                         max_features=25000)
# text_embeddings = model.fit_transform(df['title']).toarray()
# print('text embeddings shape', text_embeddings.shape)
#
# KNN = 50
# if len(df) == 3:
#     KNN = 2
# model = NearestNeighbors(n_neighbors=KNN)
# model.fit(text_embeddings)
#
# preds = []
# CHUNK = 1024 * 4
# THRESHOLD = 0.5
#
# print('Finding similar texts...')
# CTS = len(text_embeddings) // CHUNK
# if len(text_embeddings) % CHUNK != 0:
#     CTS += 1
#
# for j in range(CTS):
#
#     a = j * CHUNK
#     b = (j + 1) * CHUNK
#     b = min(b, len(text_embeddings))
#     print('chunk', a, 'to', b)
#
#     cts = cupy.matmul(text_embeddings, text_embeddings[a:b].T).T
#
#     for k in range(b - a):
#         IDX = cupy.where(cts[k,] > THRESHOLD)[0]
#         o = df.iloc[cupy.asnumpy(IDX)].posting_id.values
#         preds.append(o)


def get_metric(col):
    def f1score(row):
        n = len(np.intersect1d(row.target, row[col]))
        return 2 * n / (len(row.target) + len(row[col]))

    return f1score


def get_preds(embs_path, threshold):
    image_embeddings = np.load(embs_path)['embeddings']

    KNN = 50
    if len(df) == 3:
        KNN = 2
    model = NearestNeighbors(n_neighbors=KNN)
    model.fit(image_embeddings)

    image_embeddings = cupy.array(image_embeddings)

    preds = []
    CHUNK = 1024 * 4

    # print('Finding similar images...')
    CTS = len(image_embeddings) // CHUNK
    if len(image_embeddings) % CHUNK != 0:
        CTS += 1

    for j in range(CTS):

        a = j * CHUNK
        b = (j + 1) * CHUNK
        b = min(b, len(image_embeddings))

        cts = cupy.matmul(image_embeddings, image_embeddings[a:b].T).T

        for k in range(b - a):
            IDX = cupy.where(cts[k, ] > threshold)[0]
            IDX = cupy.asnumpy(IDX)
            o = cpu_df.iloc[IDX].posting_id.values
            preds.append(o)

    return preds


r = re.compile(r'epoch(\d+)_')

results = {}

for i, embs_path in enumerate(sorted(glob.iglob('image/embs/*.npz'), key=lambda x: int(r.search(x).group(0).replace('epoch', '').replace('_', '')))):
    print(embs_path)
    results[i] = []

    for thresh in np.arange(0.5, 1.001, 0.05):
        predictions = get_preds(embs_path, thresh)
        cpu_df['predictions'] = predictions

        cpu_df['f1'] = cpu_df.apply(get_metric('predictions'), axis=1)
        f1_mean = cpu_df['f1'].mean()
        print('\tTH = {}\tCV score = {}'.format(round(thresh, 3), round(f1_mean, 5)))
        results[i].append(f1_mean)

result_df = pd.DataFrame(data=results, index=np.arange(0.5, 1.001, 0.05))
sns.lineplot(data=result_df)
plt.savefig('image/th_plot.png')
