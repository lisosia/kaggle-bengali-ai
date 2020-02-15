import gc
import time

import six
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data.dataset import Dataset

import config

C = config.get_config()

#####################################################################
# Data load
#####################################################################
# cache as feather format
# pickle may be faster? https://blog.amedama.jp/entry/2018/07/11/081050

def prepare_image_test(datadir, featherdir, data_type, submission=False, indices=[0, 1, 2, 3]):
    assert data_type in ['train', 'test']
    print("prepare_image() start")
    if submission:  # read from parquet when submission
        image_df_list = [pd.read_parquet(datadir / f'{data_type}_image_data_{i}.parquet')
                         for i in indices]
    else:  # read from feather for speed
        image_df_list = [pd.read_feather(featherdir / f'{data_type}_image_data_{i}.feather')
                         for i in indices]

    print('image_df_list', len(image_df_list))
    HEIGHT = 137
    WIDTH = 236
    images = [df.iloc[:, 1:].values.reshape(-1, HEIGHT, WIDTH) for df in image_df_list]
    del image_df_list
    gc.collect()
    images = np.concatenate(images, axis=0)
    print("prepare_image() end")
    return images

def prepare_image(datadir, featherdir, data_type='train',
                  submission=False, indices=[0, 1, 2, 3]):
    assert data_type in ['train', 'test']
    if data_type == 'test':  # for test, same as before
        return prepare_image_test(datadir, featherdir, data_type, submission, indices)
    
    print("prepare_image() start")
    
    HEIGHT = 137
    WIDTH = 236

    PER_INDEX = 50210
    images_all = np.zeros((PER_INDEX * len(indices), 137, 236), dtype=np.uint8)
    print("    allocated:", images_all.shape)
    for i in indices:
        if submission:  # read from parquet when submission
            df = pd.read_parquet(datadir / f'{data_type}_image_data_{i}.parquet')
        else:
            df = pd.read_feather(featherdir / f'{data_type}_image_data_{i}.feather')
            
        assert len(df) == PER_INDEX
        images_all[PER_INDEX*i: PER_INDEX*(i+1)] = df.iloc[:, 1:].values.reshape(-1, HEIGHT, WIDTH).astype(np.uint8)
        del df
        gc.collect()

        print(f'    indice {i} done')

    gc.collect()
    print("prepare_image() end")
    return images_all

#####################################################################
# Dataset
#####################################################################
"""
Referenced `chainer.dataset.DatasetMixin` to work with pytorch Dataset.
"""

class DatasetMixin(Dataset):

    def __init__(self, transform=None):
        self.transform = transform

    def __getitem__(self, index):
        """Returns an example or a sequence of examples."""
        if torch.is_tensor(index):
            index = index.tolist()
        if isinstance(index, slice):
            current, stop, step = index.indices(len(self))
            return [self.get_example_wrapper(i) for i in
                    six.moves.range(current, stop, step)]
        elif isinstance(index, list) or isinstance(index, np.ndarray):
            return [self.get_example_wrapper(i) for i in index]
        else:
            return self.get_example_wrapper(index)

    def __len__(self):
        """Returns the number of data points."""
        raise NotImplementedError

    def get_example_wrapper(self, i):
        """Wrapper of `get_example`, to apply `transform` if necessary"""
        example = self.get_example(i)
        if self.transform:
            example = self.transform(example)
        return example

    def get_example(self, i):
        """Returns the i-th example.

        Implementations should override it. It should raise :class:`IndexError`
        if the index is invalid.

        Args:
            i (int): The index of the example.

        Returns:
            The i-th example.

        """
        raise NotImplementedError


class BengaliAIDataset(DatasetMixin):
    def __init__(self, images, labels=None, transform=None, indices=None):
        super(BengaliAIDataset, self).__init__(transform=transform)
        self.images = images
        if labels is not None:
            self.labels = labels.astype(np.int64)
        if indices is None:
            indices = np.arange(len(images))
        self.indices = indices
        self.train = labels is not None

    def __len__(self):
        """return length of this dataset"""
        return len(self.indices)

    def get_example(self, i):
        """Return i-th data"""
        i = self.indices[i]
        x = self.images[i]
        # Opposite white and black: background will be white and
        # for future Affine transformation
        x = (255 - x).astype(np.float32) / 255.
        if self.train:
            y = self.labels[i]
            return x, y
        else:
            return x

def get_onehot(index, num_class):
    e = np.zeros(num_class, np.float)
    e[index] = 1.
    return e

class BengaliAIDatasetPNG(DatasetMixin):
    def __init__(self, image_ids, labels=None, transform=None):
        super(BengaliAIDatasetPNG, self).__init__(transform=transform)
        self.image_ids = image_ids
        self.labels = labels
        self.train = labels is not None

    def __len__(self):
        """return length of this dataset"""
        return len(self.image_ids)

    def get_example(self, i):
        """Return i-th data"""
        x = cv2.imread(
            str(C.pngdir) + f'/{self.image_ids[i]}.png', 
            cv2.IMREAD_GRAYSCALE)
        # Opposite white and black: background will be white and
        # for future Affine transformation
        x = (255 - x).astype(np.float32) / 255.
        if self.train:
            y = self.labels[i]
            return x, y
        else:
            return x


#####################################################################
# Data Aug
#####################################################################
"""
From https://www.kaggle.com/corochann/deep-learning-cnn-with-chainer-lb-0-99700
"""
import cv2
from skimage.transform import AffineTransform, warp
import numpy as np

def affine_image(img):
    """
    Args:
        img: (h, w) or (1, h, w)

    Returns:
        img: (h, w)
    """
    # ch, h, w = img.shape
    # img = img / 255.
    if img.ndim == 3:
        img = img[0]

    # --- scale ---
    min_scale = 0.8
    max_scale = 1.2
    sx = np.random.uniform(min_scale, max_scale)
    sy = np.random.uniform(min_scale, max_scale)

    # --- rotation ---
    max_rot_angle = 7
    rot_angle = np.random.uniform(-max_rot_angle, max_rot_angle) * np.pi / 180.

    # --- shear ---
    max_shear_angle = 10
    shear_angle = np.random.uniform(-max_shear_angle, max_shear_angle) * np.pi / 180.

    # --- translation ---
    max_translation = 4
    tx = np.random.randint(-max_translation, max_translation)
    ty = np.random.randint(-max_translation, max_translation)

    tform = AffineTransform(scale=(sx, sy), rotation=rot_angle, shear=shear_angle,
                            translation=(tx, ty))
    transformed_image = warp(img, tform)
    assert transformed_image.ndim == 2
    return transformed_image

def _bbox(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, rmax, cmin, cmax

# def crop_char_image2(img0, size=SIZE, pad=16):
def crop_char_image2(img0, pad=16):
    HEIGHT = 137
    WIDTH = 236

    #crop a box around pixels large than the threshold 
    #some images contain line at the sides
    ymin,ymax,xmin,xmax = _bbox(img0[5:-5,5:-5] > 80 / 255.)
    #cropping may cut too much, so we need to add it back
    xmin = xmin - 13 if (xmin > 13) else 0
    ymin = ymin - 10 if (ymin > 10) else 0
    xmax = xmax + 13 if (xmax < WIDTH - 13) else WIDTH
    ymax = ymax + 10 if (ymax < HEIGHT - 10) else HEIGHT
    img = img0[ymin:ymax,xmin:xmax]
    #remove lo intensity pixels as noise
    img[img < (28 / 255.)] = 0
    lx, ly = xmax-xmin,ymax-ymin
    l = max(lx,ly) + pad
    #make sure that the aspect ratio is kept in rescaling
    img = np.pad(img, [((l-ly)//2,), ((l-lx)//2,)], mode='constant')
    return img
    # return cv2.resize(img,(size,size))

#def crop_char_image(image, threshold=5./255.):
#    assert image.ndim == 2
#    is_black = image > threshold
#
#    is_black_vertical = np.sum(is_black, axis=0) > 0
#    is_black_horizontal = np.sum(is_black, axis=1) > 0
#    left = np.argmax(is_black_horizontal)
#    right = np.argmax(is_black_horizontal[::-1])
#    top = np.argmax(is_black_vertical)
#    bottom = np.argmax(is_black_vertical[::-1])
#    height, width = image.shape
#    cropped_image = image[left:height - right, top:width - bottom]
#    return cropped_image

def resize(image, size=(128, 128)):  # size is (H,W)
    return cv2.resize(image, (size[1], size[0]))

### [Update] I added albumentations augmentations introduced in Bengali: albumentations data augmentation tutorial.
import albumentations as A
import numpy as np


def add_gaussian_noise(x, sigma):
    x += np.random.randn(*x.shape) * sigma
    x = np.clip(x, 0., 1.)
    return x


def _evaluate_ratio(ratio):
    if ratio <= 0.:
        return False
    return np.random.uniform() < ratio


def apply_aug(aug, image):
    return aug(image=image)['image']


class Transform:
    def __init__(self, affine=True, size=(64, 64),
                 normalize=False, train=True, threshold=40.,
                 sigma=-1., blur_ratio=0., noise_ratio=0., cutout_ratio=0.,
                 grid_distortion_ratio=0., elastic_distortion_ratio=0., random_brightness_ratio=0.,
                 piece_affine_ratio=0., ssr_ratio=0.):
        self.affine = affine
        self.size = size
        self.normalize = normalize
        self.train = train
        self.threshold = threshold / 255.
        self.sigma = sigma / 255.

        self.blur_ratio = blur_ratio
        self.noise_ratio = noise_ratio
        self.cutout_ratio = cutout_ratio
        self.grid_distortion_ratio = grid_distortion_ratio
        self.elastic_distortion_ratio = elastic_distortion_ratio
        self.random_brightness_ratio = random_brightness_ratio
        self.piece_affine_ratio = piece_affine_ratio
        self.ssr_ratio = ssr_ratio

    def __call__(self, example):
        if self.train:
            x, y = example
        else:
            x = example

        # --- Train/Test common preprocessing ---
        x = crop_char_image2(x / x.max(), pad=C.crop_pad_size)

        # --- Augmentation ---
        if self.affine:
            x = apply_aug(A.ShiftScaleRotate(
                    shift_limit=(4./C.image_size[0]), scale_limit=tuple(C.aug_scale), rotate_limit=7,
                    border_mode=cv2.BORDER_CONSTANT, value=0., p=1.0),
                    x)
            ## comment out. should rotate around center
            #    x = affine_image(x)

        # --- Train/Test common preprocessing ---
        if self.size is not None:
            x = resize(x, size=self.size)  # H, W

        if self.sigma > 0.:
            x = add_gaussian_noise(x, sigma=self.sigma)

        # albumentations...
        x = x.astype(np.float32)  # use float
        assert x.ndim == 2
        # 1. blur
        if _evaluate_ratio(self.blur_ratio):
            r = np.random.uniform()
            if r < 0.25:
                x = apply_aug(A.Blur(p=1.0), x)
            elif r < 0.5:
                x = apply_aug(A.MedianBlur(blur_limit=5, p=1.0), x)
            elif r < 0.75:
                x = apply_aug(A.GaussianBlur(p=1.0), x)
            else:
                x = apply_aug(A.MotionBlur(p=1.0), x)

        # 2. noise
        if _evaluate_ratio(self.noise_ratio):
            r = np.random.uniform()
            if r < 0.50:
                x = apply_aug(A.GaussNoise(var_limit=5. / 255., p=1.0), x)
            else:
                x = apply_aug(A.MultiplicativeNoise(p=1.0), x)  # 乗算ノイズ

        # Cutout
        if _evaluate_ratio(self.cutout_ratio):
            # A.Cutout(num_holes=2,  max_h_size=2, max_w_size=2, p=1.0)  # Deprecated...
            x = apply_aug(A.CoarseDropout(max_holes=8, max_height=8, max_width=8, p=1.0), x)

        if _evaluate_ratio(self.grid_distortion_ratio):
            x = apply_aug(A.GridDistortion(p=1.0), x)

        if _evaluate_ratio(self.elastic_distortion_ratio):
            x = apply_aug(A.ElasticTransform(
                sigma=50, alpha=1, alpha_affine=10, p=1.0), x)

        if _evaluate_ratio(self.random_brightness_ratio):
            # A.RandomBrightness(p=1.0)  # Deprecated...
            # A.RandomContrast(p=1.0)    # Deprecated...
            x = apply_aug(A.RandomBrightnessContrast(p=1.0), x)

        if _evaluate_ratio(self.piece_affine_ratio):
            x = apply_aug(A.IAAPiecewiseAffine(p=1.0), x)

        if _evaluate_ratio(self.ssr_ratio):
            x = apply_aug(A.ShiftScaleRotate(
                shift_limit=0.0625,
                scale_limit=0.1,
                rotate_limit=30,
                p=1.0), x)

        if self.normalize:
            x = (x.astype(np.float32) - 0.0692) / 0.2051
        if x.ndim == 2:
            x = x[None, :, :]
        x = x.astype(np.float32)
        if self.train:
            y = y.astype(np.int64)
            return x, y
        else:
            return x


# load data
train = pd.read_csv(C.datadir/'train.csv')
train_labels = train[['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']].values

def get_trainval_dataset():
    _time_start = time.time()
    indices = [0, 1, 2, 3]
    # indices = [0, 1, 2]
    train_images = prepare_image(
        C.datadir, C.featherdir, data_type='train', submission=False, indices=indices)
    print(f'data load time was {time.time() - _time_start}')

    _df_fold = pd.read_csv('../input/train_with_fold.csv')
    _fold_train = np.where(_df_fold.fold.values != str(C.fold))
    _fold_valid = np.where(_df_fold.fold.values == str(C.fold))
    print(f'pre-split fold loaded, valid_indices:{_fold_valid}')
    assert _fold_train.size + _fold_valid.size == len(train_images)
    
    train_dataset = BengaliAIDataset(
        train_images, train_labels,
        transform=Transform(size=C.image_size),
        indices=_fold_train)
    valid_dataset = BengaliAIDataset(
        train_images, train_labels,
        transform=Transform(affine=False, size=C.image_size),
        indices=_fold_valid)

    print('train_dataset', len(train_dataset), 'valid_dataset', len(valid_dataset))
    return train_dataset, valid_dataset

from sklearn.model_selection import train_test_split
def get_trainval_dataset_png():
    print(train.image_id.values.shape)
    print(train_labels.shape)
    
    _df_fold = pd.read_csv('../input/train_with_fold.csv')
    print(_df_fold.head())
    _fold_train = np.where(_df_fold.fold.values != C.fold)
    _fold_valid = np.where(_df_fold.fold.values == C.fold)
    print(f'pre-split fold loaded, valid_indices:{_fold_valid}')
    assert _fold_train[0].size + _fold_valid[0].size == len(train.image_id.values)

    x_train = train.image_id.values[_fold_train]
    y_train = train_labels[_fold_train]
    x_valid = train.image_id.values[_fold_valid]
    y_valid = train_labels[_fold_valid]
    
    train_dataset = BengaliAIDatasetPNG(
        x_train, y_train, 
        transform=Transform(size=C.image_size))
    valid_dataset = BengaliAIDatasetPNG(
        x_valid, y_valid, 
        transform=Transform(affine=False, size=C.image_size))
    print('train_dataset', len(train_dataset), 'valid_dataset', len(valid_dataset))

    return train_dataset, valid_dataset


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    train_dataset = BengaliAIDataset(train_images, train_labels)

    if False:  # affine_image test
        nrow, ncol = 1, 6

        fig, axes = plt.subplots(nrow, ncol, figsize=(20, 8))
        axes = axes.flatten()
        for i, ax in tqdm(enumerate(axes)):
            image, label = train_dataset[0]
            ax.imshow(affine_image(image), cmap='Greys')
            ax.set_title(f'label: {label}')
        plt.tight_layout()

    # crop test
    if False:
        nrow, ncol = 5, 6

        fig, axes = plt.subplots(nrow, ncol, figsize=(20, 8))
        axes = axes.flatten()
        for i, ax in tqdm(enumerate(axes)):
            image, label = train_dataset[i]
            ax.imshow(crop_char_image(image, threshold=20./255.), cmap='Greys')
            ax.set_title(f'label: {label}')
        plt.tight_layout()


    # crop & resize test
    if False:
        nrow, ncol = 5, 6

        fig, axes = plt.subplots(nrow, ncol, figsize=(20, 8))
        axes = axes.flatten()
        for i, ax in tqdm(enumerate(axes)):
            image, label = train_dataset[i]
            ax.imshow(resize(crop_char_image(image, threshold=20./255.)), cmap='Greys')
            ax.set_title(f'label: {label}')
        plt.tight_layout()


    # visualize
    if True:
        image, label = train_dataset[0]
        print('image', image.shape, 'label', label)
        nrow, ncol = 5, 6
        fig, axes = plt.subplots(nrow, ncol, figsize=(20, 8))
        axes = axes.flatten()
        for i, ax in tqdm(enumerate(axes)):
            image, label = train_dataset[i]
            ax.imshow(image, cmap='Greys')
            ax.set_title(f'label: {label}')
        plt.tight_layout()
        plt.show()
        plt.savefig('bengaliai.png')


    if True:  # test augmentation by albmentation
        nrow, ncol = 1, 6

        fig, axes = plt.subplots(nrow, ncol, figsize=(20, 2))
        axes = axes.flatten()
        for i, ax in tqdm(enumerate(axes)):
            image, label = train_dataset[0]
            ax.imshow(image[0], cmap='Greys')
            ax.set_title(f'label: {label}')
        plt.tight_layout()


    ### Final check for train dataset
    if True:
        nrow, ncol = 5, 6

        fig, axes = plt.subplots(nrow, ncol, figsize=(20, 8))
        axes = axes.flatten()
        for i, ax in tqdm(enumerate(axes)):
            image, label = train_dataset[i]
            ax.imshow(image[0], cmap='Greys')
            ax.set_title(f'label: {label}')
        plt.tight_layout()
