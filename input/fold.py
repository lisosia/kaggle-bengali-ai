import pandas as pd
import numpy as np

ids = pd.read_csv('../input/bengaliai-cv19/train.csv').image_id

n_dataset = len(ids)
train_data_size = int(n_dataset * 0.9)
valid_data_size = int(n_dataset - train_data_size)
print(f'total:{n_dataset} train:{train_data_size} valid:{valid_data_size}')
perm = np.random.RandomState(777).permutation(n_dataset)
train_dx = perm[:train_data_size]
valid_dx = perm[train_data_size:]
assert valid_dx.size == valid_data_size

fold = np.empty(n_dataset, dtype=object)
print(fold)
fold[train_dx] = 'train'
fold[valid_dx] = 'valid'

df = pd.DataFrame({'fold': fold})
df.to_csv('fold_trainval.csv', index=True)