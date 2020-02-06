import gc

import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm

HEIGHT = 137
WIDTH = 236
# SIZE = 128

TRAIN = ['../input/bengaliai-cv19/train_image_data_0.parquet',
         '../input/bengaliai-cv19/train_image_data_1.parquet',
         '../input/bengaliai-cv19/train_image_data_2.parquet',
         '../input/bengaliai-cv19/train_image_data_3.parquet']

OUTDIR = '../input/bengaliai-cv19-png/'

if False:  # test
    df = pd.read_parquet(TRAIN[0])
    n_imgs = 8
    fig, axs = plt.subplots(n_imgs, 2, figsize=(10, 5*n_imgs))

    for idx in range(n_imgs):
        #somehow the original input is inverted
        img0 = 255 - df.iloc[idx, 1:].values.reshape(HEIGHT, WIDTH).astype(np.uint8)
        #normalize each image by its max val
        img = (img0*(255.0/img0.max())).astype(np.uint8)
        img = crop_resize(img)

        axs[idx,0].imshow(img0)
        axs[idx,0].set_title('Original image')
        axs[idx,0].axis('off')
        axs[idx,1].imshow(img)
        axs[idx,1].set_title('Crop & resize')
        axs[idx,1].axis('off')
    plt.show()

for fname in TRAIN:
    df = pd.read_parquet(fname)
    #note: the input is inverted
    data = df.iloc[:, 1:].values.reshape(-1, HEIGHT, WIDTH).astype(np.uint8)
    for idx in tqdm(range(len(df))):
        name = df.iloc[idx,0]
        cv2.imwrite(
            OUTDIR + name + ".png",
            data[idx]
        )
    del df
    del data
    gc.collect()
