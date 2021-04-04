import cudf
import cupy
import pandas as pd
import numpy as np
from cuml.feature_extraction.text import TfidfVectorizer
from cuml.neighbors import NearestNeighbors

from sklearn.metrics import f1_score

print('Computing text embeddings...')


def get_embeddings_metric(df):
    for i in range(len(df)):
        df.loc[df.loc[i, 'preds'], 'pred_label'] = df.loc[i, 'label_group']
    f1 = f1_score(df['label_group'], df['pred_label'], average='macro')
    return f1


cpu_df = pd.read_csv('image/train.csv')
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


image_embeddings = np.load('image/embs_effnet4.npz')['embeddings']
print('text embeddings shape', image_embeddings.shape)

KNN = 50
if len(df) == 3:
    KNN = 2
model = NearestNeighbors(n_neighbors=KNN)
model.fit(image_embeddings)

image_embeddings = cupy.array(image_embeddings)

for threshold in np.arange(0.5, 1.1, 0.1):
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
        # print('chunk', a, 'to', b)

        cts = cupy.matmul(image_embeddings, image_embeddings[a:b].T).T

        for k in range(b - a):
            IDX = cupy.where(cts[k, ] > threshold)[0]
            IDX = cupy.asnumpy(IDX)
            o = cpu_df.iloc[IDX].index.values
            preds.append(o)
        pass

    cpu_df['preds'] = preds

    print(threshold, get_embeddings_metric(cpu_df))
