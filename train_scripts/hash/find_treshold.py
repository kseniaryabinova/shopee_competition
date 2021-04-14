import pandas as pd
import numpy as np

from pandarallel import pandarallel
pandarallel.initialize()


df = pd.read_csv('../../dataset/train_fold.csv')

tmp = df.groupby('label_group').posting_id.agg('unique').to_dict()
df['target'] = df.label_group.map(tmp)

df['hash_bin'] = df['image_phash'].apply(lambda x: "{0:08b}".format(int(x, 16)))


def hamming_distance(chaine1, chaine2):
    return sum(c1 != c2 for c1, c2 in zip(chaine1, chaine2))


def get_simular_hashes(orig_hash, thresh):
    simular_hash_ids = []
    for i, single_hash in enumerate(df['hash_bin']):
        if hamming_distance(orig_hash, single_hash) < thresh:
            simular_hash_ids.append(df.loc[i]['posting_id'])
    return simular_hash_ids


def get_metric(col):
    def f1score(row):
        n = len(np.intersect1d(row.target, row[col]))
        return 2 * n / (len(row.target) + len(row[col]))

    return f1score


for th in range(1, 64):
    df['preds'] = df['hash_bin'].parallel_apply(lambda x: get_simular_hashes(x, th))

    df['f1'] = df.apply(get_metric('preds'), axis=1)
    print(th, df['f1'].mean())
