import cudf
import cupy
import pandas as pd
import numpy as np
from cuml.feature_extraction.text import TfidfVectorizer
from cuml.neighbors import NearestNeighbors

print('Computing text embeddings...')

cpu_df = pd.read_csv('train.csv')
cpu_df = pd.concat([cpu_df, cpu_df], axis=0)
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



image_embeddings = np.load('embs0.npz')['embeddings']
image_embeddings = np.concatenate([image_embeddings, image_embeddings], axis=0)
print('text embeddings shape', image_embeddings.shape)

KNN = 50
if len(df) == 3:
    KNN = 2
model = NearestNeighbors(n_neighbors=KNN)
model.fit(image_embeddings)

image_embeddings = cupy.array(image_embeddings)
preds = []
CHUNK = 1024 * 4
THRESHOLD = 1

print('Finding similar images...')
CTS = len(image_embeddings) // CHUNK
if len(image_embeddings) % CHUNK != 0:
    CTS += 1

for j in range(CTS):

    a = j * CHUNK
    b = (j + 1) * CHUNK
    b = min(b, len(image_embeddings))
    print('chunk', a, 'to', b)

    cts = cupy.matmul(image_embeddings, image_embeddings[a:b].T).T

    for k in range(b - a):
        IDX = cupy.where(cts[k,] > THRESHOLD)[0]
        IDX = cupy.asnumpy(IDX)
        o = cpu_df.iloc[IDX].posting_id.values
        preds.append(o)

print(len(preds))
