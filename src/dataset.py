import gc
import time

import six
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data.dataset import Dataset

import config
C = config.load_config()

#####################################################################
# Data load
#####################################################################
# cache as feather format
# pickle may be faster? https://blog.amedama.jp/entry/2018/07/11/081050
def prepare_image(datadir, featherdir, data_type='train',
                  submission=False, indices=[0, 1, 2, 3]):
    assert data_type in ['train', 'test']
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
    return images

# load data
_time_start = time.time()
train = pd.read_csv(C.datadir/'train.csv')
train_labels = train[['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']].values
indices = [0] if C.debug else [0, 1, 2, 3]
train_images = prepare_image(
    C.datadir, C.featherdir, data_type='train', submission=False, indices=indices)
print(f'data load time was {time.time() - _time_start}')


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
        self.labels = labels
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


def crop_char_image(image, threshold=5./255.):
    assert image.ndim == 2
    is_black = image > threshold

    is_black_vertical = np.sum(is_black, axis=0) > 0
    is_black_horizontal = np.sum(is_black, axis=1) > 0
    left = np.argmax(is_black_horizontal)
    right = np.argmax(is_black_horizontal[::-1])
    top = np.argmax(is_black_vertical)
    bottom = np.argmax(is_black_vertical[::-1])
    height, width = image.shape
    cropped_image = image[left:height - right, top:width - bottom]
    return cropped_image


def resize(image, size=(128, 128)):
    return cv2.resize(image, size)

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
    def __init__(self, affine=True, crop=True, size=(64, 64),
                 normalize=True, train=True, threshold=40.,
                 sigma=-1., blur_ratio=0., noise_ratio=0., cutout_ratio=0.,
                 grid_distortion_ratio=0., elastic_distortion_ratio=0., random_brightness_ratio=0.,
                 piece_affine_ratio=0., ssr_ratio=0.):
        self.affine = affine
        self.crop = crop
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
        # --- Augmentation ---
        if self.affine:
            x = affine_image(x)

        # --- Train/Test common preprocessing ---
        if self.crop:
            x = crop_char_image(x, threshold=self.threshold)
        if self.size is not None:
            x = resize(x, size=self.size)
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


def get_train_dataset_justfortest():
    train_transform = Transform(
        size=(image_size, image_size), threshold=20.,
        sigma=-1., blur_ratio=0.2, noise_ratio=0.2, cutout_ratio=0.2,
        grid_distortion_ratio=0.2, random_brightness_ratio=0.2,
        piece_affine_ratio=0.2, ssr_ratio=0.2)
    return BengaliAIDataset(train_images, train_labels,
                                 transform=train_transform)

def get_trainval_dataset():
    n_dataset = len(train_images)
    train_data_size = 200 if C.debug else int(n_dataset * 0.9)
    valid_data_size = 100 if C.debug else int(n_dataset - train_data_size)

    perm = np.random.RandomState(777).permutation(n_dataset)
    print('perm', perm)
    train_dataset = BengaliAIDataset(
        train_images, train_labels, transform=Transform(size=(C.image_size, C.image_size)),
        indices=perm[:train_data_size])
    valid_dataset = BengaliAIDataset(
        train_images, train_labels, transform=Transform(affine=False, crop=True, size=(C.image_size, C.image_size)),
        indices=perm[train_data_size:train_data_size+valid_data_size])
    print('train_dataset', len(train_dataset), 'valid_dataset', len(valid_dataset))

    return train_dataset, valid_dataset


########## COPYED FROM PREDICTION KERNEL #############
# note. treth is 20, image_size is 128 for all 4 provided models
## transform_test = Transform(affine=False, crop=True, size=(image_size, image_size), threshold, train=False)
transform_test = Transform(affine=False, crop=True, size=(C.image_size, C.image_size), threshold=20, train=False)
########## COPYED FROM PREDICTION KERNEL #############

#def get_test_dataset(idx):
#    """idx must be one of [1, 2, 3, 4]"""


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    train_dataset = BengaliAIDataset(train_images, train_labels)

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
