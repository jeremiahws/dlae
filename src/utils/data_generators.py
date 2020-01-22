# Copyright 2019 Jeremiah Sanders.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
'''src/utils/data_generators_2D.py

Data generators for 2D data.
'''


from src.utils.data_utils import *
from src.utils.general_utils import *
from src.utils.ssd_utils import BoxFilter
from random import shuffle
from scipy import linalg
import h5py


class CNN2DDatasetGenerator(object):
    def __init__(self,
                 imgs_hdf5_path,
                 annos_hdf5_path=None,
                 rotation_range=0,
                 width_shift_range=0.,
                 height_shift_range=0.,
                 shear_range=0.,
                 zoom_range=0.,
                 flip_horizontal=False,
                 flip_vertical=False,
                 featurewise_center=False,
                 featurewise_std_normalization=False,
                 samplewise_center=False,
                 samplewise_std_normalization=False,
                 zca_epsilon=None,
                 brightness_range=None,
                 channel_shift_range=0.1,
                 shuffle_data=False,
                 rounds=1,
                 fill_mode='nearest',
                 cval=0,
                 interpolation_order=1,
                 seed=None,
                 batch_size=4,
                 validation_split=0.0,
                 subset='train',
                 normalization=None,
                 min_intensity=[0.],
                 max_intensity=[0.],
                 categorical_labels=False,
                 num_classes=None,
                 repeat_chans=False,
                 chan_repititions=0,
                 apply_aug=False):
        self.imgs_hdf5_path = imgs_hdf5_path
        self.annos_hdf5_path = annos_hdf5_path
        self.batch_size = batch_size
        self.keys = get_keys(self.imgs_hdf5_path)
        self.keys.sort(key=int)
        self.shuffle_data = shuffle_data
        self.rotation_range = rotation_range
        self.width_shift_range = width_shift_range
        self.height_shift_range = height_shift_range
        self.shear_range = shear_range
        self.zoom_range = [1 - zoom_range, 1 + zoom_range]
        self.flip_horizontal = flip_horizontal
        self.flip_vertical = flip_vertical
        self.featurewise_center = featurewise_center
        self.featurewise_std_normalization = featurewise_std_normalization
        self.samplewise_center = samplewise_center
        self.samplewise_std_normalization = samplewise_std_normalization
        self.zca_epsilon = zca_epsilon
        self.brightness_range = brightness_range
        self.channel_shift_range = channel_shift_range
        self.fill_mode = fill_mode
        self.cval = cval
        self.interpolation_order = interpolation_order
        self.rounds = rounds
        self.seed = seed
        self.validation_split = validation_split
        self.subset = subset
        if self.subset == 'train' or self.subset == 'validation':
            self.proceed = check_keys(self.imgs_hdf5_path, self.annos_hdf5_path)
        else:
            self.proceed = None
        self.normalization = normalization
        self.min_intensity = min_intensity
        self.max_intensity = max_intensity
        self.to_categorical = categorical_labels
        self.num_classes = num_classes
        if self.subset == 'train':
            if self.shuffle_data:
                shuffle(self.keys)
            self.train_dataset_size = np.floor(len(self.keys) * (1 - self.validation_split))
            self.test_dataset_size = None
            self.train_dataset_indices = np.arange(self.train_dataset_size, dtype=np.int32)
            self.test_dataset_indices = None
            if self.validation_split > 0.:
                self.val_dataset_size = np.ceil(len(self.keys) * self.validation_split)
                self.val_dataset_indices = np.arange(self.val_dataset_size, dtype=np.int32)
                split_idx = int(np.ceil(len(self.keys) * self.validation_split))
                self.validation_keys = self.keys[:split_idx]
                self.train_keys = self.keys[split_idx:]
            else:
                self.val_dataset_size = None
                self.val_dataset_indices = None
                self.validation_keys = None
                self.train_keys = self.keys
            self.test_keys = None
        elif self.subset == 'validation':
            self.train_dataset_size = None
            self.val_dataset_size = len(self.keys)
            self.test_dataset_size = None
            self.train_dataset_indices = None
            self.val_dataset_indices = np.arange(self.val_dataset_size, dtype=np.int32)
            self.test_dataset_indices = None
            self.train_keys = None
            self.validation_keys = self.keys
            self.test_keys = None
        elif self.subset == 'test':
            self.train_dataset_size = None
            self.val_dataset_size = None
            self.test_dataset_size = len(self.keys)
            self.train_dataset_indices = None
            self.val_dataset_indices = None
            self.test_dataset_indices = np.arange(self.test_dataset_size, dtype=np.int32)
            self.train_keys = None
            self.validation_keys = None
            self.test_keys = self.keys
        else:
            raise ValueError('Invalid subset specified. Valid values are train, validation, or test.')
        self.repeat_chans = repeat_chans
        self.chan_repititions = chan_repititions
        self.apply_augmentation = apply_aug

    def __len__(self):
        """
        The "length" of the generator is the number of batches expected.
        :return: the expected number of batches that will be produced by this generator.
        """
        if self.subset == 'train':
            return int(np.ceil(self.rounds * self.train_dataset_size / self.batch_size))
        elif self.subset == 'validation':
            if self.val_dataset_size is None:
                raise ValueError('Zero validation data was reserved. If you want to generate a validation set,'
                                 + ' set 0 < validation_split < 1')
            else:
                return int(np.ceil(self.val_dataset_size / self.batch_size))
        elif self.subset == 'test':
            return int(np.ceil(self.test_dataset_size / self.batch_size))
        else:
            raise ValueError('Invalidation subset defined. Only train, validation, and test are valid subsets.')

    def generate(self):
        """
        Reads in data from an HDF5 file, applies augmentation chain (if
        desired), shuffles and batches the data.
        """
        row_axis = 0
        col_axis = 1
        chan_axis = 2
        if self.proceed is False:
            raise ValueError('Datset names in the X (image) and y (annotation) HDF5 files are not identical.'
                             + ' Images and annotations must be paired and have identical names in the HDF5 files.')

        if self.seed is not None:
            np.random.seed(self.seed)

        current_train = 0
        current_val = 0
        current_test = 0

        while True:
            if self.subset == 'train':
                batch_X, batch_y = [], []

                if current_train >= self.train_dataset_size:
                    current_train = 0

                    if self.shuffle_data:
                        shuffle(self.train_keys)

                batch_indices = self.train_dataset_indices[current_train:current_train + self.batch_size]
                f_imgs = h5py.File(self.imgs_hdf5_path, 'r')
                f_annos = h5py.File(self.annos_hdf5_path, 'r')

                for i in batch_indices:
                    sample_name = self.train_keys[i]
                    img = f_imgs[sample_name].value
                    if len(img.shape) == 2:
                        img = np.expand_dims(img, axis=-1)
                    anno = f_annos[sample_name].value
                    anno = anno.astype('float32')
                    if self.to_categorical:
                        orig_classes = np.unique(anno)
                    else:
                        orig_classes = None
                    orig_shape = img.shape

                    if self.repeat_chans:
                        img = np.repeat(img, self.chan_repititions, axis=-1)

                    if self.featurewise_center:
                        mean = np.mean(img, axis=(0, row_axis, col_axis))
                        broadcast_shape = [1, 1, 1]
                        broadcast_shape[chan_axis - 1] = img.shape[chan_axis]
                        mean = np.reshape(mean, broadcast_shape)
                        img -= mean

                    if self.featurewise_std_normalization:
                        std = np.std(img, axis=(0, row_axis, col_axis))
                        broadcast_shape = [1, 1, 1]
                        broadcast_shape[chan_axis - 1] = img.shape[chan_axis]
                        std = np.reshape(std, broadcast_shape)
                        img /= (std + 1e-6)

                    if self.samplewise_center:
                        img -= np.mean(img, keepdims=True)

                    if self.samplewise_std_normalization:
                        img /= (np.std(img, keepdims=True) + 1e-6)

                    if self.zca_epsilon is not None and self.apply_augmentation is True:
                        flat_x = np.reshape(
                            img, (img.shape[0], np.prod(img.shape[1:])))
                        sigma = np.dot(flat_x.T, flat_x) / flat_x.shape[0]
                        u, s, _ = linalg.svd(sigma)
                        s_inv = 1. / np.sqrt(s[np.newaxis] + self.zca_epsilon)
                        principal_components = (u * s_inv).dot(u.T)
                        flatx = np.reshape(img, (-1, np.prod(img.shape[1:])))
                        whitex = np.dot(flatx, principal_components)
                        img = np.reshape(whitex, img.shape)

                    if self.rotation_range:
                        theta = np.random.uniform(-self.rotation_range, self.rotation_range)
                    else:
                        theta = 0

                    if self.height_shift_range:
                        try:  # 1-D array-like or int
                            tx = np.random.choice(self.height_shift_range)
                            tx *= np.random.choice([-1, 1])
                        except ValueError:  # floating point
                            tx = np.random.uniform(-self.height_shift_range,
                                                   self.height_shift_range)
                        if np.max(self.height_shift_range) < 1:
                            tx *= img.shape[row_axis]
                    else:
                        tx = 0

                    if self.width_shift_range:
                        try:  # 1-D array-like or int
                            ty = np.random.choice(self.width_shift_range)
                            ty *= np.random.choice([-1, 1])
                        except ValueError:  # floating point
                            ty = np.random.uniform(-self.width_shift_range,
                                                   self.width_shift_range)
                        if np.max(self.width_shift_range) < 1:
                            ty *= img.shape[col_axis]
                    else:
                        ty = 0

                    if self.shear_range:
                        shear = np.random.uniform(
                            -self.shear_range,
                            self.shear_range)
                    else:
                        shear = 0

                    if self.zoom_range[0] == 1 and self.zoom_range[1] == 1:
                        zx, zy = 1, 1
                    else:
                        zx, zy = np.random.uniform(
                            self.zoom_range[0],
                            self.zoom_range[1],
                            2)

                    brightness = None
                    if self.brightness_range is not None:
                        if orig_shape[-1] == 1:
                            img = np.repeat(img, 3, axis=-1)
                        brightness = np.random.uniform(self.brightness_range[0],
                                                       self.brightness_range[1])

                    channel_shift_intensity = None
                    if self.channel_shift_range != 0:
                        channel_shift_intensity = np.random.uniform(-self.channel_shift_range,
                                                                    self.channel_shift_range)

                    flip_horizontal = (np.random.random() < 0.5) * self.flip_horizontal
                    flip_vertical = (np.random.random() < 0.5) * self.flip_vertical

                    transform_parameters_img = {'theta': theta,
                                                'tx': tx,
                                                'ty': ty,
                                                'shear': shear,
                                                'zx': zx,
                                                'zy': zy,
                                                'flip_horizontal': flip_horizontal,
                                                'flip_vertical': flip_vertical,
                                                'fill_mode': self.fill_mode,
                                                'cval': self.cval,
                                                'interpolation_order': self.interpolation_order,
                                                'brightness': brightness,
                                                'channel_shift_intensity': channel_shift_intensity}

                    if self.apply_augmentation is True:
                        img = apply_transform(img, transform_parameters_img)

                    if self.normalization is not None:
                        if self.normalization == 'samplewise_unity_x':
                            if (np.max(img) - np.min(img)) != 0:
                                img = (img - np.min(img)) / (np.max(img) - np.min(img))
                        elif self.normalization == 'samplewise_negpos_x':
                            if (np.max(img) - np.min(img)) != 0:
                                img = 2 * (((img - np.min(img)) / (np.max(img) - np.min(img))) - 0.5)
                        elif self.normalization == 'global_unity_x':
                            if (self.max_intensity[0] - self.min_intensity[0]) != 0:
                                img = (img - self.min_intensity[0]) / (self.max_intensity[0] - self.min_intensity[0])
                        elif self.normalization == 'global_negpos_x':
                            if (self.max_intensity[0] - self.min_intensity[0]) != 0:
                                img = 2 * (((img - self.min_intensity[0]) / (self.max_intensity[0] - self.min_intensity[0])) - 0.5)
                        elif self.normalization == 'samplewise_unity_xy':
                            if (np.max(img) - np.min(img)) != 0:
                                img = (img - np.min(img)) / (np.max(img) - np.min(img))
                        elif self.normalization == 'samplewise_negpos_xy':
                            if (np.max(img) - np.min(img)) != 0:
                                img = 2 * (((img - np.min(img)) / (np.max(img) - np.min(img))) - 0.5)
                        elif self.normalization == 'global_unity_xy':
                            if (self.max_intensity[0] - self.min_intensity[0]) != 0:
                                img = (img - self.min_intensity[0]) / (self.max_intensity[0] - self.min_intensity[0])
                        elif self.normalization == 'global_negpos_xy':
                            if (self.max_intensity[0] - self.min_intensity[0]) != 0:
                                img = 2 * (((img - self.min_intensity[0]) / (self.max_intensity[0] - self.min_intensity[0])) - 0.5)
                        elif self.normalization == 'none':
                            pass
                        else:
                            raise ValueError('Normalization type must be either samplewise_unity_x,'
                                             + ' samplewise_negpos_x, global_unity_x, global_negpos_x,'
                                             + ' samplewise_unity_xy, samplewise_negpos_xy, global_unity_xy,'
                                             + ' global_negpos_xy, or none.')

                    if orig_shape[-1] == 1:
                        img = np.take(img, 0, axis=-1)
                        img = np.expand_dims(img, axis=-1)

                    batch_X.append(img)
                    if self.to_categorical:
                        if self.num_classes is None:
                            raise ValueError('If converting to categorical variables, you must specify the number'
                                             + ' of classes.')
                        else:
                            integers = np.isin(anno, orig_classes)
                            anno = np.multiply(anno, integers)
                            anno = to_categorical(anno, num_classes=self.num_classes)
                    batch_y.append(anno)

                f_imgs.close()
                f_annos.close()

                current_train += self.batch_size

                batch_X = np.array(batch_X)
                batch_y = np.array(batch_y)
                if len(batch_y.shape) == 3:
                    batch_y = np.squeeze(batch_y, axis=1)
                ret = (batch_X, batch_y)

                yield ret

            elif self.subset == 'validation':
                if self.val_dataset_size is None:
                    raise ValueError('Zero validation data was reserved. If you want to generate a validation set,'
                                      + ' set 0 < validation_split < 1')
                else:
                    batch_X, batch_y = [], []

                    if current_val >= self.val_dataset_size:
                        current_val = 0

                    batch_indices = self.val_dataset_indices[current_val:current_val + self.batch_size]
                    f_imgs = h5py.File(self.imgs_hdf5_path, 'r')
                    f_annos = h5py.File(self.annos_hdf5_path, 'r')
                    for i in batch_indices:
                        sample_name = self.validation_keys[i]
                        img = f_imgs[sample_name].value
                        if len(img.shape) == 2:
                            img = np.expand_dims(img, axis=-1)
                        anno = f_annos[sample_name].value
                        anno = anno.astype('float32')
                        if self.to_categorical:
                            orig_classes = np.unique(anno)
                        else:
                            orig_classes = None

                        if self.repeat_chans:
                            img = np.repeat(img, self.chan_repititions, axis=-1)

                        if self.featurewise_center:
                            mean = np.mean(img, axis=(0, row_axis, col_axis))
                            broadcast_shape = [1, 1, 1]
                            broadcast_shape[chan_axis - 1] = img.shape[chan_axis]
                            mean = np.reshape(mean, broadcast_shape)
                            img -= mean

                        if self.featurewise_std_normalization:
                            std = np.std(img, axis=(0, row_axis, col_axis))
                            broadcast_shape = [1, 1, 1]
                            broadcast_shape[chan_axis - 1] = img.shape[chan_axis]
                            std = np.reshape(std, broadcast_shape)
                            img /= (std + 1e-6)

                        if self.samplewise_center:
                            img -= np.mean(img, keepdims=True)

                        if self.samplewise_std_normalization:
                            img /= (np.std(img, keepdims=True) + 1e-6)

                        if self.normalization is not None:
                            if self.normalization == 'samplewise_unity_x':
                                if (np.max(img) - np.min(img)) != 0:
                                    img = (img - np.min(img)) / (np.max(img) - np.min(img))
                            elif self.normalization == 'samplewise_negpos_x':
                                if (np.max(img) - np.min(img)) != 0:
                                    img = 2 * (((img - np.min(img)) / (np.max(img) - np.min(img))) - 0.5)
                            elif self.normalization == 'global_unity_x':
                                if (self.max_intensity[0] - self.min_intensity[0]) != 0:
                                    img = (img - self.min_intensity[0]) / (self.max_intensity[0] - self.min_intensity[0])
                            elif self.normalization == 'global_negpos_x':
                                if (self.max_intensity[0] - self.min_intensity[0]) != 0:
                                    img = 2 * (((img - self.min_intensity[0]) / (self.max_intensity[0] - self.min_intensity[0])) - 0.5)
                            elif self.normalization == 'samplewise_unity_xy':
                                if (np.max(img) - np.min(img)) != 0:
                                    img = (img - np.min(img)) / (np.max(img) - np.min(img))
                            elif self.normalization == 'samplewise_negpos_xy':
                                if (np.max(img) - np.min(img)) != 0:
                                    img = 2 * (((img - np.min(img)) / (np.max(img) - np.min(img))) - 0.5)
                            elif self.normalization == 'global_unity_xy':
                                if (self.max_intensity[0] - self.min_intensity[0]) != 0:
                                    img = (img - self.min_intensity[0]) / (self.max_intensity[0] - self.min_intensity[0])
                            elif self.normalization == 'global_negpos_xy':
                                if (self.max_intensity[0] - self.min_intensity[0]) != 0:
                                    img = 2 * (((img - self.min_intensity[0]) / (self.max_intensity[0] - self.min_intensity[0])) - 0.5)
                            elif self.normalization == 'none':
                                pass
                            else:
                                raise ValueError('Normalization type must be either samplewise_unity_x,'
                                                 + ' samplewise_negpos_x, global_unity_x, global_negpos_x,'
                                                 + ' samplewise_unity_xy, samplewise_negpos_xy, global_unity_xy,'
                                                 + ' global_negpos_xy, or none.')

                        batch_X.append(img)
                        if self.to_categorical:
                            if self.num_classes is None:
                                raise ValueError('If converting to categorical variables, you must specify the number'
                                                 + ' of classes.')
                            else:
                                integers = np.isin(anno, orig_classes)
                                anno = np.multiply(anno, integers)
                                anno = to_categorical(anno, num_classes=self.num_classes)
                        batch_y.append(anno)

                    f_imgs.close()
                    f_annos.close()

                    current_val += self.batch_size

                    batch_X = np.array(batch_X)
                    batch_y = np.array(batch_y)
                    if len(batch_y.shape) == 3:
                        batch_y = np.squeeze(batch_y, axis=1)
                    ret = (batch_X, batch_y)

                    yield ret

            elif self.subset == 'test':
                batch_X = []

                if current_test >= self.test_dataset_size:
                    current_test = 0

                batch_indices = self.test_dataset_indices[current_test:current_test + self.batch_size]
                f_imgs = h5py.File(self.imgs_hdf5_path, 'r')
                for i in batch_indices:
                    sample_name = self.test_keys[i]
                    img = f_imgs[sample_name].value
                    if len(img.shape) == 2:
                        img = np.expand_dims(img, axis=-1)

                    if self.repeat_chans:
                        img = np.repeat(img, self.chan_repititions, axis=-1)

                    if self.normalization is not None:
                        if self.normalization == 'samplewise_unity_x':
                            if (np.max(img) - np.min(img)) != 0:
                                img = (img - np.min(img)) / (np.max(img) - np.min(img))
                        elif self.normalization == 'samplewise_negpos_x':
                            if (np.max(img) - np.min(img)) != 0:
                                img = 2 * (((img - np.min(img)) / (np.max(img) - np.min(img))) - 0.5)
                        elif self.normalization == 'global_unity_x':
                            if (self.max_intensity[0] - self.min_intensity[0]) != 0:
                                img = (img - self.min_intensity[0]) / (self.max_intensity[0] - self.min_intensity[0])
                        elif self.normalization == 'global_negpos_x':
                            if (self.max_intensity[0] - self.min_intensity[0]) != 0:
                                img = 2 * (((img - self.min_intensity[0]) / (self.max_intensity[0] - self.min_intensity[0])) - 0.5)
                        elif self.normalization == 'samplewise_unity_xy':
                            if (np.max(img) - np.min(img)) != 0:
                                img = (img - np.min(img)) / (np.max(img) - np.min(img))
                        elif self.normalization == 'samplewise_negpos_xy':
                            if (np.max(img) - np.min(img)) != 0:
                                img = 2 * (((img - np.min(img)) / (np.max(img) - np.min(img))) - 0.5)
                        elif self.normalization == 'global_unity_xy':
                            if (self.max_intensity[0] - self.min_intensity[0]) != 0:
                                img = (img - self.min_intensity[0]) / (self.max_intensity[0] - self.min_intensity[0])
                        elif self.normalization == 'global_negpos_xy':
                            if (self.max_intensity[0] - self.min_intensity[0]) != 0:
                                img = 2 * (((img - self.min_intensity[0]) / (self.max_intensity[0] - self.min_intensity[0])) - 0.5)
                        elif self.normalization == 'none':
                            pass
                        else:
                            raise ValueError('Normalization type must be either samplewise_unity_x,'
                                             + ' samplewise_negpos_x, global_unity_x, global_negpos_x,'
                                             + ' samplewise_unity_xy, samplewise_negpos_xy, global_unity_xy,'
                                             + ' global_negpos_xy, or none.')

                    batch_X.append(img)

                f_imgs.close()

                current_test += self.batch_size

                batch_X = np.array(batch_X)
                ret = (batch_X)

                yield ret


class CNN3DDatasetGenerator(object):
    def __init__(self,
                 imgs_hdf5_path,
                 annos_hdf5_path=None,
                 rotation_range=0,
                 width_shift_range=0.,
                 height_shift_range=0.,
                 shear_range=0.,
                 zoom_range=0.,
                 flip_horizontal=False,
                 flip_vertical=False,
                 featurewise_center=False,
                 featurewise_std_normalization=False,
                 samplewise_center=False,
                 samplewise_std_normalization=False,
                 zca_epsilon=None,
                 brightness_range=None,
                 channel_shift_range=0.1,
                 shuffle_data=False,
                 rounds=1,
                 fill_mode='nearest',
                 cval=0,
                 interpolation_order=1,
                 seed=None,
                 batch_size=4,
                 validation_split=0.0,
                 subset='train',
                 normalization=None,
                 min_intensity=[0.],
                 max_intensity=[0.],
                 categorical_labels=False,
                 num_classes=None,
                 repeat_chans=False,
                 chan_repititions=0,
                 apply_aug=False):
        self.imgs_hdf5_path = imgs_hdf5_path
        self.annos_hdf5_path = annos_hdf5_path
        self.batch_size = batch_size
        self.keys = get_keys(self.imgs_hdf5_path)
        self.keys.sort(key=int)
        self.shuffle_data = shuffle_data
        self.rotation_range = rotation_range
        self.width_shift_range = width_shift_range
        self.height_shift_range = height_shift_range
        self.shear_range = shear_range
        self.zoom_range = [1 - zoom_range, 1 + zoom_range]
        self.flip_horizontal = flip_horizontal
        self.flip_vertical = flip_vertical
        self.featurewise_center = featurewise_center
        self.featurewise_std_normalization = featurewise_std_normalization
        self.samplewise_center = samplewise_center
        self.samplewise_std_normalization = samplewise_std_normalization
        self.zca_epsilon = zca_epsilon
        self.brightness_range = brightness_range
        self.channel_shift_range = channel_shift_range
        self.fill_mode = fill_mode
        self.cval = cval
        self.interpolation_order = interpolation_order
        self.rounds = rounds
        self.seed = seed
        self.validation_split = validation_split
        self.subset = subset
        if self.subset == 'train' or self.subset == 'validation':
            self.proceed = check_keys(self.imgs_hdf5_path, self.annos_hdf5_path)
        else:
            self.proceed = None
        self.normalization = normalization
        self.min_intensity = min_intensity
        self.max_intensity = max_intensity
        self.to_categorical = categorical_labels
        self.num_classes = num_classes
        if self.subset == 'train':
            if self.shuffle_data:
                shuffle(self.keys)
            self.train_dataset_size = np.floor(len(self.keys) * (1 - self.validation_split))
            self.test_dataset_size = None
            self.train_dataset_indices = np.arange(self.train_dataset_size, dtype=np.int32)
            self.test_dataset_indices = None
            if self.validation_split > 0.:
                self.val_dataset_size = np.ceil(len(self.keys) * self.validation_split)
                self.val_dataset_indices = np.arange(self.val_dataset_size, dtype=np.int32)
                split_idx = int(np.ceil(len(self.keys) * self.validation_split))
                self.validation_keys = self.keys[:split_idx]
                self.train_keys = self.keys[split_idx:]
            else:
                self.val_dataset_size = None
                self.val_dataset_indices = None
                self.validation_keys = None
                self.train_keys = self.keys
            self.test_keys = None
        elif self.subset == 'validation':
            self.train_dataset_size = None
            self.val_dataset_size = len(self.keys)
            self.test_dataset_size = None
            self.train_dataset_indices = None
            self.val_dataset_indices = np.arange(self.val_dataset_size, dtype=np.int32)
            self.test_dataset_indices = None
            self.train_keys = None
            self.validation_keys = self.keys
            self.test_keys = None
        elif self.subset == 'test':
            self.train_dataset_size = None
            self.val_dataset_size = None
            self.test_dataset_size = len(self.keys)
            self.train_dataset_indices = None
            self.val_dataset_indices = None
            self.test_dataset_indices = np.arange(self.test_dataset_size, dtype=np.int32)
            self.train_keys = None
            self.validation_keys = None
            self.test_keys = self.keys
        else:
            raise ValueError('Invalid subset specified. Valid values are train, validation, or test.')
        self.repeat_chans = repeat_chans
        self.chan_repititions = chan_repititions
        self.apply_augmentation = apply_aug

    def __len__(self):
        """
        The "length" of the generator is the number of batches expected.
        :return: the expected number of batches that will be produced by this generator.
        """
        if self.subset == 'train':
            return int(np.ceil(self.rounds * self.train_dataset_size / self.batch_size))
        elif self.subset == 'validation':
            if self.val_dataset_size is None:
                raise ValueError('Zero validation data was reserved. If you want to generate a validation set,'
                                 + ' set 0 < validation_split < 1')
            else:
                return int(np.ceil(self.val_dataset_size / self.batch_size))
        elif self.subset == 'test':
            return int(np.ceil(self.test_dataset_size / self.batch_size))
        else:
            raise ValueError('Invalidation subset defined. Only train, validation, and test are valid subsets.')

    def generate(self):
        """
        Reads in data from an HDF5 file, applies augmentation chain (if
        desired), shuffles and batches the data.
        """
        row_axis = 0
        col_axis = 1
        slice_axis = 2
        chan_axis = 3
        if self.proceed is False:
            raise ValueError('Datset names in the X (image) and y (annotation) HDF5 files are not identical.'
                             + ' Images and annotations must be paired and have identical names in the HDF5 files.')

        if self.seed is not None:
            np.random.seed(self.seed)

        current_train = 0
        current_val = 0
        current_test = 0

        while True:
            if self.subset == 'train':
                batch_X, batch_y = [], []

                if current_train >= self.train_dataset_size:
                    current_train = 0

                    if self.shuffle_data:
                        shuffle(self.train_keys)

                batch_indices = self.train_dataset_indices[current_train:current_train + self.batch_size]
                f_imgs = h5py.File(self.imgs_hdf5_path, 'r')
                f_annos = h5py.File(self.annos_hdf5_path, 'r')

                for i in batch_indices:
                    sample_name = self.train_keys[i]
                    img = f_imgs[sample_name].value
                    if len(img.shape) == 3:
                        img = np.expand_dims(img, axis=-1)
                    anno = f_annos[sample_name].value
                    anno = anno.astype('float32')
                    if self.to_categorical:
                        orig_classes = np.unique(anno)
                    else:
                        orig_classes = None
                    orig_shape = img.shape

                    if self.repeat_chans:
                        img = np.repeat(img, self.chan_repititions, axis=-1)

                    if self.featurewise_center:
                        mean = np.mean(img, axis=(0, row_axis, col_axis, slice_axis))
                        broadcast_shape = [1, 1, 1, 1]
                        broadcast_shape[chan_axis - 1] = img.shape[chan_axis]
                        mean = np.reshape(mean, broadcast_shape)
                        img -= mean

                    if self.featurewise_std_normalization:
                        std = np.std(img, axis=(0, row_axis, col_axis, slice_axis))
                        broadcast_shape = [1, 1, 1, 1]
                        broadcast_shape[chan_axis - 1] = img.shape[chan_axis]
                        std = np.reshape(std, broadcast_shape)
                        img /= (std + 1e-6)

                    if self.samplewise_center:
                        img -= np.mean(img, keepdims=True)

                    if self.samplewise_std_normalization:
                        img /= (np.std(img, keepdims=True) + 1e-6)

                    if self.zca_epsilon is not None and self.apply_augmentation is True:
                        flat_x = np.reshape(
                            img, (img.shape[0], np.prod(img.shape[1:])))
                        sigma = np.dot(flat_x.T, flat_x) / flat_x.shape[0]
                        u, s, _ = linalg.svd(sigma)
                        s_inv = 1. / np.sqrt(s[np.newaxis] + self.zca_epsilon)
                        principal_components = (u * s_inv).dot(u.T)
                        flatx = np.reshape(img, (-1, np.prod(img.shape[1:])))
                        whitex = np.dot(flatx, principal_components)
                        img = np.reshape(whitex, img.shape)

                    if self.rotation_range:
                        theta = np.random.uniform(-self.rotation_range, self.rotation_range)
                    else:
                        theta = 0

                    if self.height_shift_range:
                        try:  # 1-D array-like or int
                            tx = np.random.choice(self.height_shift_range)
                            tx *= np.random.choice([-1, 1])
                        except ValueError:  # floating point
                            tx = np.random.uniform(-self.height_shift_range,
                                                   self.height_shift_range)
                        if np.max(self.height_shift_range) < 1:
                            tx *= img.shape[row_axis]
                    else:
                        tx = 0

                    if self.width_shift_range:
                        try:  # 1-D array-like or int
                            ty = np.random.choice(self.width_shift_range)
                            ty *= np.random.choice([-1, 1])
                        except ValueError:  # floating point
                            ty = np.random.uniform(-self.width_shift_range,
                                                   self.width_shift_range)
                        if np.max(self.width_shift_range) < 1:
                            ty *= img.shape[col_axis]
                    else:
                        ty = 0

                    if self.shear_range:
                        shear = np.random.uniform(
                            -self.shear_range,
                            self.shear_range)
                    else:
                        shear = 0

                    if self.zoom_range[0] == 1 and self.zoom_range[1] == 1:
                        zx, zy = 1, 1
                    else:
                        zx, zy = np.random.uniform(
                            self.zoom_range[0],
                            self.zoom_range[1],
                            2)

                    brightness = None
                    if self.brightness_range is not None:
                        if orig_shape[-1] == 1:
                            img = np.repeat(img, 3, axis=-1)
                        brightness = np.random.uniform(self.brightness_range[0],
                                                       self.brightness_range[1])

                    channel_shift_intensity = None
                    if self.channel_shift_range != 0:
                        channel_shift_intensity = np.random.uniform(-self.channel_shift_range,
                                                                    self.channel_shift_range)

                    flip_horizontal = (np.random.random() < 0.5) * self.flip_horizontal
                    flip_vertical = (np.random.random() < 0.5) * self.flip_vertical

                    transform_parameters_img = {'theta': theta,
                                                'tx': tx,
                                                'ty': ty,
                                                'shear': shear,
                                                'zx': zx,
                                                'zy': zy,
                                                'flip_horizontal': flip_horizontal,
                                                'flip_vertical': flip_vertical,
                                                'fill_mode': self.fill_mode,
                                                'cval': self.cval,
                                                'interpolation_order': self.interpolation_order,
                                                'brightness': brightness,
                                                'channel_shift_intensity': channel_shift_intensity}

                    if self.apply_augmentation is True:
                        new_img = []
                        for i in range(img.shape[2]):
                            aug = img[:, :, i, :]
                            aug = apply_transform(aug, transform_parameters_img)
                            new_img.append(aug)
                        img = np.array(new_img)
                        img = np.transpose(img, (1, 2, 0, 3))

                    if self.normalization is not None:
                        if self.normalization == 'samplewise_unity_x':
                            if (np.max(img) - np.min(img)) != 0:
                                img = (img - np.min(img)) / (np.max(img) - np.min(img))
                        elif self.normalization == 'samplewise_negpos_x':
                            if (np.max(img) - np.min(img)) != 0:
                                img = 2 * (((img - np.min(img)) / (np.max(img) - np.min(img))) - 0.5)
                        elif self.normalization == 'global_unity_x':
                            if (self.max_intensity[0] - self.min_intensity[0]) != 0:
                                img = (img - self.min_intensity[0]) / (self.max_intensity[0] - self.min_intensity[0])
                        elif self.normalization == 'global_negpos_x':
                            if (self.max_intensity[0] - self.min_intensity[0]) != 0:
                                img = 2 * (((img - self.min_intensity[0]) / (self.max_intensity[0] - self.min_intensity[0])) - 0.5)
                        elif self.normalization == 'samplewise_unity_xy':
                            if (np.max(img) - np.min(img)) != 0:
                                img = (img - np.min(img)) / (np.max(img) - np.min(img))
                        elif self.normalization == 'samplewise_negpos_xy':
                            if (np.max(img) - np.min(img)) != 0:
                                img = 2 * (((img - np.min(img)) / (np.max(img) - np.min(img))) - 0.5)
                        elif self.normalization == 'global_unity_xy':
                            if (self.max_intensity[0] - self.min_intensity[0]) != 0:
                                img = (img - self.min_intensity[0]) / (self.max_intensity[0] - self.min_intensity[0])
                        elif self.normalization == 'global_negpos_xy':
                            if (self.max_intensity[0] - self.min_intensity[0]) != 0:
                                img = 2 * (((img - self.min_intensity[0]) / (self.max_intensity[0] - self.min_intensity[0])) - 0.5)
                        elif self.normalization == 'none':
                            pass
                        else:
                            raise ValueError('Normalization type must be either samplewise_unity_x,'
                                             + ' samplewise_negpos_x, global_unity_x, global_negpos_x,'
                                             + ' samplewise_unity_xy, samplewise_negpos_xy, global_unity_xy,'
                                             + ' global_negpos_xy, or none.')

                    batch_X.append(img)
                    if self.to_categorical:
                        if self.num_classes is None:
                            raise ValueError('If converting to categorical variables, you must specify the number'
                                             + ' of classes.')
                        else:
                            integers = np.isin(anno, orig_classes)
                            anno = np.multiply(anno, integers)
                            anno = to_categorical(anno, num_classes=self.num_classes)
                    batch_y.append(anno)

                f_imgs.close()
                f_annos.close()

                current_train += self.batch_size

                batch_X = np.array(batch_X)
                batch_y = np.array(batch_y)
                if len(batch_y.shape) == 3:
                    batch_y = np.squeeze(batch_y, axis=1)
                ret = (batch_X, batch_y)

                yield ret

            elif self.subset == 'validation':
                if self.val_dataset_size is None:
                    raise ValueError('Zero validation data was reserved. If you want to generate a validation set,'
                                      + ' set 0 < validation_split < 1')
                else:
                    batch_X, batch_y = [], []

                    if current_val >= self.val_dataset_size:
                        current_val = 0

                    batch_indices = self.val_dataset_indices[current_val:current_val + self.batch_size]
                    f_imgs = h5py.File(self.imgs_hdf5_path, 'r')
                    f_annos = h5py.File(self.annos_hdf5_path, 'r')
                    for i in batch_indices:
                        sample_name = self.validation_keys[i]
                        img = f_imgs[sample_name].value
                        if len(img.shape) == 3:
                            img = np.expand_dims(img, axis=-1)
                        anno = f_annos[sample_name].value
                        anno = anno.astype('float32')
                        if self.to_categorical:
                            orig_classes = np.unique(anno)
                        else:
                            orig_classes = None

                        if self.repeat_chans:
                            img = np.repeat(img, self.chan_repititions, axis=-1)

                        if self.featurewise_center:
                            mean = np.mean(img, axis=(0, row_axis, col_axis, slice_axis))
                            broadcast_shape = [1, 1, 1, 1]
                            broadcast_shape[chan_axis - 1] = img.shape[chan_axis]
                            mean = np.reshape(mean, broadcast_shape)
                            img -= mean

                        if self.featurewise_std_normalization:
                            std = np.std(img, axis=(0, row_axis, col_axis, slice_axis))
                            broadcast_shape = [1, 1, 1, 1]
                            broadcast_shape[chan_axis - 1] = img.shape[chan_axis]
                            std = np.reshape(std, broadcast_shape)
                            img /= (std + 1e-6)

                        if self.samplewise_center:
                            img -= np.mean(img, keepdims=True)

                        if self.samplewise_std_normalization:
                            img /= (np.std(img, keepdims=True) + 1e-6)

                        if self.normalization is not None:
                            if self.normalization == 'samplewise_unity_x':
                                if (np.max(img) - np.min(img)) != 0:
                                    img = (img - np.min(img)) / (np.max(img) - np.min(img))
                            elif self.normalization == 'samplewise_negpos_x':
                                if (np.max(img) - np.min(img)) != 0:
                                    img = 2 * (((img - np.min(img)) / (np.max(img) - np.min(img))) - 0.5)
                            elif self.normalization == 'global_unity_x':
                                if (self.max_intensity[0] - self.min_intensity[0]) != 0:
                                    img = (img - self.min_intensity[0]) / (self.max_intensity[0] - self.min_intensity[0])
                            elif self.normalization == 'global_negpos_x':
                                if (self.max_intensity[0] - self.min_intensity[0]) != 0:
                                    img = 2 * (((img - self.min_intensity[0]) / (self.max_intensity[0] - self.min_intensity[0])) - 0.5)
                            elif self.normalization == 'samplewise_unity_xy':
                                if (np.max(img) - np.min(img)) != 0:
                                    img = (img - np.min(img)) / (np.max(img) - np.min(img))
                            elif self.normalization == 'samplewise_negpos_xy':
                                if (np.max(img) - np.min(img)) != 0:
                                    img = 2 * (((img - np.min(img)) / (np.max(img) - np.min(img))) - 0.5)
                            elif self.normalization == 'global_unity_xy':
                                if (self.max_intensity[0] - self.min_intensity[0]) != 0:
                                    img = (img - self.min_intensity[0]) / (self.max_intensity[0] - self.min_intensity[0])
                            elif self.normalization == 'global_negpos_xy':
                                if (self.max_intensity[0] - self.min_intensity[0]) != 0:
                                    img = 2 * (((img - self.min_intensity[0]) / (self.max_intensity[0] - self.min_intensity[0])) - 0.5)
                            elif self.normalization == 'none':
                                pass
                            else:
                                raise ValueError('Normalization type must be either samplewise_unity_x,'
                                                 + ' samplewise_negpos_x, global_unity_x, global_negpos_x,'
                                                 + ' samplewise_unity_xy, samplewise_negpos_xy, global_unity_xy,'
                                                 + ' global_negpos_xy, or none.')

                        if orig_shape[-1] == 1:
                            img = np.take(img, 0, axis=-1)
                            img = np.expand_dims(img, axis=-1)

                        batch_X.append(img)
                        if self.to_categorical:
                            if self.num_classes is None:
                                raise ValueError('If converting to categorical variables, you must specify the number'
                                                 + ' of classes.')
                            else:
                                integers = np.isin(anno, orig_classes)
                                anno = np.multiply(anno, integers)
                                anno = to_categorical(anno, num_classes=self.num_classes)
                        batch_y.append(anno)

                    f_imgs.close()
                    f_annos.close()

                    current_val += self.batch_size

                    batch_X = np.array(batch_X)
                    batch_y = np.array(batch_y)
                    if len(batch_y.shape) == 3:
                        batch_y = np.squeeze(batch_y, axis=1)
                    ret = (batch_X, batch_y)

                    yield ret

            elif self.subset == 'test':
                batch_X = []

                if current_test >= self.test_dataset_size:
                    current_test = 0

                batch_indices = self.test_dataset_indices[current_test:current_test + self.batch_size]
                f_imgs = h5py.File(self.imgs_hdf5_path, 'r')
                for i in batch_indices:
                    sample_name = self.test_keys[i]
                    img = f_imgs[sample_name].value
                    if len(img.shape) == 3:
                        img = np.expand_dims(img, axis=-1)

                    if self.repeat_chans:
                        img = np.repeat(img, self.chan_repititions, axis=-1)

                    if self.normalization is not None:
                        if self.normalization == 'samplewise_unity_x':
                            if (np.max(img) - np.min(img)) != 0:
                                img = (img - np.min(img)) / (np.max(img) - np.min(img))
                        elif self.normalization == 'samplewise_negpos_x':
                            if (np.max(img) - np.min(img)) != 0:
                                img = 2 * (((img - np.min(img)) / (np.max(img) - np.min(img))) - 0.5)
                        elif self.normalization == 'global_unity_x':
                            if (self.max_intensity[0] - self.min_intensity[0]) != 0:
                                img = (img - self.min_intensity[0]) / (self.max_intensity[0] - self.min_intensity[0])
                        elif self.normalization == 'global_negpos_x':
                            if (self.max_intensity[0] - self.min_intensity[0]) != 0:
                                img = 2 * (((img - self.min_intensity[0]) / (self.max_intensity[0] - self.min_intensity[0])) - 0.5)
                        elif self.normalization == 'samplewise_unity_xy':
                            if (np.max(img) - np.min(img)) != 0:
                                img = (img - np.min(img)) / (np.max(img) - np.min(img))
                        elif self.normalization == 'samplewise_negpos_xy':
                            if (np.max(img) - np.min(img)) != 0:
                                img = 2 * (((img - np.min(img)) / (np.max(img) - np.min(img))) - 0.5)
                        elif self.normalization == 'global_unity_xy':
                            if (self.max_intensity[0] - self.min_intensity[0]) != 0:
                                img = (img - self.min_intensity[0]) / (self.max_intensity[0] - self.min_intensity[0])
                        elif self.normalization == 'global_negpos_xy':
                            if (self.max_intensity[0] - self.min_intensity[0]) != 0:
                                img = 2 * (((img - self.min_intensity[0]) / (self.max_intensity[0] - self.min_intensity[0])) - 0.5)
                        elif self.normalization == 'none':
                            pass
                        else:
                            raise ValueError('Normalization type must be either samplewise_unity_x,'
                                             + ' samplewise_negpos_x, global_unity_x, global_negpos_x,'
                                             + ' samplewise_unity_xy, samplewise_negpos_xy, global_unity_xy,'
                                             + ' global_negpos_xy, or none.')

                    batch_X.append(img)

                f_imgs.close()

                current_test += self.batch_size

                batch_X = np.array(batch_X)
                ret = (batch_X)

                yield ret


class FCN2DDatasetGenerator(object):
    def __init__(self,
                 imgs_hdf5_path,
                 annos_hdf5_path=None,
                 rotation_range=0,
                 width_shift_range=0.,
                 height_shift_range=0.,
                 shear_range=0.,
                 zoom_range=0.,
                 flip_horizontal=False,
                 flip_vertical=False,
                 featurewise_center=False,
                 featurewise_std_normalization=False,
                 samplewise_center=False,
                 samplewise_std_normalization=False,
                 zca_epsilon=None,
                 brightness_range=None,
                 channel_shift_range=0.1,
                 shuffle_data=False,
                 rounds=1,
                 fill_mode='nearest',
                 cval=0,
                 interpolation_order=1,
                 seed=None,
                 batch_size=4,
                 validation_split=0.0,
                 subset='train',
                 normalization=None,
                 min_intensity=[0.],
                 max_intensity=[0.],
                 categorical_labels=False,
                 num_classes=None,
                 repeat_chans=False,
                 chan_repititions=0,
                 apply_aug=False):
        self.imgs_hdf5_path = imgs_hdf5_path
        self.annos_hdf5_path = annos_hdf5_path
        self.batch_size = batch_size
        self.keys = get_keys(self.imgs_hdf5_path)
        self.keys.sort(key=int)
        self.shuffle_data = shuffle_data
        self.rotation_range = rotation_range
        self.width_shift_range = width_shift_range
        self.height_shift_range = height_shift_range
        self.shear_range = shear_range
        self.zoom_range = [1 - zoom_range, 1 + zoom_range]
        self.flip_horizontal = flip_horizontal
        self.flip_vertical = flip_vertical
        self.featurewise_center = featurewise_center
        self.featurewise_std_normalization = featurewise_std_normalization
        self.samplewise_center = samplewise_center
        self.samplewise_std_normalization = samplewise_std_normalization
        self.zca_epsilon = zca_epsilon
        self.brightness_range = brightness_range
        self.channel_shift_range = channel_shift_range
        self.fill_mode = fill_mode
        self.cval = cval
        self.interpolation_order = interpolation_order
        self.rounds = rounds
        self.seed = seed
        self.validation_split = validation_split
        self.subset = subset
        if self.subset == 'train' or self.subset == 'validation':
            self.proceed = check_keys(self.imgs_hdf5_path, self.annos_hdf5_path)
        else:
            self.proceed = None
        self.normalization = normalization
        self.min_intensity = min_intensity
        self.max_intensity = max_intensity
        self.to_categorical = categorical_labels
        self.num_classes = num_classes
        if self.subset == 'train':
            if self.shuffle_data:
                shuffle(self.keys)
            self.train_dataset_size = np.floor(len(self.keys) * (1 - self.validation_split))
            self.test_dataset_size = None
            self.train_dataset_indices = np.arange(self.train_dataset_size, dtype=np.int32)
            self.test_dataset_indices = None
            if self.validation_split > 0.:
                self.val_dataset_size = np.ceil(len(self.keys) * self.validation_split)
                self.val_dataset_indices = np.arange(self.val_dataset_size, dtype=np.int32)
                split_idx = int(np.ceil(len(self.keys) * self.validation_split))
                self.validation_keys = self.keys[:split_idx]
                self.train_keys = self.keys[split_idx:]
            else:
                self.val_dataset_size = None
                self.val_dataset_indices = None
                self.validation_keys = None
                self.train_keys = self.keys
            self.test_keys = None
        elif self.subset == 'validation':
            self.train_dataset_size = None
            self.val_dataset_size = len(self.keys)
            self.test_dataset_size = None
            self.train_dataset_indices = None
            self.val_dataset_indices = np.arange(self.val_dataset_size, dtype=np.int32)
            self.test_dataset_indices = None
            self.train_keys = None
            self.validation_keys = self.keys
            self.test_keys = None
        elif self.subset == 'test':
            self.train_dataset_size = None
            self.val_dataset_size = None
            self.test_dataset_size = len(self.keys)
            self.train_dataset_indices = None
            self.val_dataset_indices = None
            self.test_dataset_indices = np.arange(self.test_dataset_size, dtype=np.int32)
            self.train_keys = None
            self.validation_keys = None
            self.test_keys = self.keys
        else:
            raise ValueError('Invalid subset specified. Valid values are train, validation, or test.')
        self.repeat_chans = repeat_chans
        self.chan_repititions = chan_repititions
        self.apply_augmentation = apply_aug

    def __len__(self):
        """
        The "length" of the generator is the number of batches expected.
        :return: the expected number of batches that will be produced by this generator.
        """
        if self.subset == 'train':
            return int(np.ceil(self.rounds * self.train_dataset_size / self.batch_size))
        elif self.subset == 'validation':
            if self.val_dataset_size is None:
                raise ValueError('Zero validation data was reserved. If you want to generate a validation set,'
                                 + ' set 0 < validation_split < 1')
            else:
                return int(np.ceil(self.val_dataset_size / self.batch_size))
        elif self.subset == 'test':
            return int(np.ceil(self.test_dataset_size / self.batch_size))
        else:
            raise ValueError('Invalidation subset defined. Only train, validation, and test are valid subsets.')

    def generate(self):
        """
        Reads in data from an HDF5 file, applies augmentation chain (if
        desired), shuffles and batches the data.
        """
        row_axis = 0
        col_axis = 1
        chan_axis = 2
        if self.proceed is False:
            raise ValueError('Datset names in the X (image) and y (annotation) HDF5 files are not identical.'
                             + ' Images and annotations must be paired and have identical names in the HDF5 files.')

        if self.seed is not None:
            np.random.seed(self.seed)

        current_train = 0
        current_val = 0
        current_test = 0

        while True:
            if self.subset == 'train':
                batch_X, batch_y = [], []

                if current_train >= self.train_dataset_size:
                    current_train = 0

                    if self.shuffle_data:
                        shuffle(self.train_keys)

                batch_indices = self.train_dataset_indices[current_train:current_train + self.batch_size]
                f_imgs = h5py.File(self.imgs_hdf5_path, 'r')
                f_annos = h5py.File(self.annos_hdf5_path, 'r')

                for i in batch_indices:
                    sample_name = self.train_keys[i]
                    img = f_imgs[sample_name].value
                    if len(img.shape) == 2:
                        img = np.expand_dims(img, axis=-1)
                    anno = f_annos[sample_name].value
                    anno = anno.astype('float32')
                    if self.to_categorical:
                        orig_classes = np.unique(anno)
                    else:
                        orig_classes = None
                    orig_shape = img.shape

                    if img.shape[0] != anno.shape[0] or img.shape[1] != anno.shape[1]:
                        raise ValueError('Images and annotations do not have the same number of rows and columns.')

                    if len(anno.shape) == 2:
                        anno = np.expand_dims(anno, axis=-1)

                    if self.repeat_chans:
                        img = np.repeat(img, self.chan_repititions, axis=-1)

                    if self.featurewise_center:
                        mean = np.mean(img, axis=(0, row_axis, col_axis))
                        broadcast_shape = [1, 1, 1]
                        broadcast_shape[chan_axis - 1] = img.shape[chan_axis]
                        mean = np.reshape(mean, broadcast_shape)
                        img -= mean

                    if self.featurewise_std_normalization:
                        std = np.std(img, axis=(0, row_axis, col_axis))
                        broadcast_shape = [1, 1, 1]
                        broadcast_shape[chan_axis - 1] = img.shape[chan_axis]
                        std = np.reshape(std, broadcast_shape)
                        img /= (std + 1e-6)

                    if self.samplewise_center:
                        img -= np.mean(img, keepdims=True)

                    if self.samplewise_std_normalization:
                        img /= (np.std(img, keepdims=True) + 1e-6)

                    if self.zca_epsilon is not None and self.apply_augmentation is True:
                        flat_x = np.reshape(
                            img, (img.shape[0], np.prod(img.shape[1:])))
                        sigma = np.dot(flat_x.T, flat_x) / flat_x.shape[0]
                        u, s, _ = linalg.svd(sigma)
                        s_inv = 1. / np.sqrt(s[np.newaxis] + self.zca_epsilon)
                        principal_components = (u * s_inv).dot(u.T)
                        flatx = np.reshape(img, (-1, np.prod(img.shape[1:])))
                        whitex = np.dot(flatx, principal_components)
                        img = np.reshape(whitex, img.shape)

                    if self.rotation_range:
                        theta = np.random.uniform(-self.rotation_range, self.rotation_range)
                    else:
                        theta = 0

                    if self.height_shift_range:
                        try:  # 1-D array-like or int
                            tx = np.random.choice(self.height_shift_range)
                            tx *= np.random.choice([-1, 1])
                        except ValueError:  # floating point
                            tx = np.random.uniform(-self.height_shift_range,
                                                   self.height_shift_range)
                        if np.max(self.height_shift_range) < 1:
                            tx *= img.shape[row_axis]
                    else:
                        tx = 0

                    if self.width_shift_range:
                        try:  # 1-D array-like or int
                            ty = np.random.choice(self.width_shift_range)
                            ty *= np.random.choice([-1, 1])
                        except ValueError:  # floating point
                            ty = np.random.uniform(-self.width_shift_range,
                                                   self.width_shift_range)
                        if np.max(self.width_shift_range) < 1:
                            ty *= img.shape[col_axis]
                    else:
                        ty = 0

                    if self.shear_range:
                        shear = np.random.uniform(
                            -self.shear_range,
                            self.shear_range)
                    else:
                        shear = 0

                    if self.zoom_range[0] == 1 and self.zoom_range[1] == 1:
                        zx, zy = 1, 1
                    else:
                        zx, zy = np.random.uniform(
                            self.zoom_range[0],
                            self.zoom_range[1],
                            2)

                    brightness = None
                    if self.brightness_range is not None:
                        if orig_shape[-1] == 1:
                            img = np.repeat(img, 3, axis=-1)
                        brightness = np.random.uniform(self.brightness_range[0],
                                                       self.brightness_range[1])

                    channel_shift_intensity = None
                    if self.channel_shift_range != 0:
                        channel_shift_intensity = np.random.uniform(-self.channel_shift_range,
                                                                    self.channel_shift_range)

                    flip_horizontal = (np.random.random() < 0.5) * self.flip_horizontal
                    flip_vertical = (np.random.random() < 0.5) * self.flip_vertical

                    transform_parameters_img = {'theta': theta,
                                                'tx': tx,
                                                'ty': ty,
                                                'shear': shear,
                                                'zx': zx,
                                                'zy': zy,
                                                'flip_horizontal': flip_horizontal,
                                                'flip_vertical': flip_vertical,
                                                'fill_mode': self.fill_mode,
                                                'cval': self.cval,
                                                'interpolation_order': self.interpolation_order,
                                                'brightness': brightness,
                                                'channel_shift_intensity': channel_shift_intensity}
                    transform_parameters_anno = {'theta': theta,
                                                 'tx': tx,
                                                 'ty': ty,
                                                 'shear': shear,
                                                 'zx': zx,
                                                 'zy': zy,
                                                 'flip_horizontal': flip_horizontal,
                                                 'flip_vertical': flip_vertical,
                                                 'fill_mode': self.fill_mode,
                                                 'cval': self.cval,
                                                 'interpolation_order': self.interpolation_order}

                    if self.apply_augmentation is True:
                        img = apply_transform(img, transform_parameters_img)
                        anno = apply_transform(anno, transform_parameters_anno)

                    if self.normalization is not None:
                        if self.normalization == 'samplewise_unity_x':
                            if (np.max(img) - np.min(img)) != 0:
                                img = (img - np.min(img)) / (np.max(img) - np.min(img))
                        elif self.normalization == 'samplewise_negpos_x':
                            if (np.max(img) - np.min(img)) != 0:
                                img = 2 * (((img - np.min(img)) / (np.max(img) - np.min(img))) - 0.5)
                        elif self.normalization == 'global_unity_x':
                            if (self.max_intensity[0] - self.min_intensity[0]) != 0:
                                img = (img - self.min_intensity[0]) / (self.max_intensity[0] - self.min_intensity[0])
                        elif self.normalization == 'global_negpos_x':
                            if (self.max_intensity[0] - self.min_intensity[0]) != 0:
                                img = 2 * (((img - self.min_intensity[0]) / (self.max_intensity[0] - self.min_intensity[0])) - 0.5)
                        elif self.normalization == 'samplewise_unity_xy':
                            if (np.max(img) - np.min(img)) != 0:
                                img = (img - np.min(img)) / (np.max(img) - np.min(img))
                            if (np.max(anno) - np.min(anno)) != 0:
                                anno = (anno - np.min(anno)) / (np.max(anno) - np.min(anno))
                        elif self.normalization == 'samplewise_negpos_xy':
                            if (np.max(img) - np.min(img)) != 0:
                                img = 2 * (((img - np.min(img)) / (np.max(img) - np.min(img))) - 0.5)
                            if (np.max(anno) - np.min(anno)) != 0:
                                anno = 2 * (((anno - np.min(anno)) / (np.max(anno) - np.min(anno))) - 0.5)
                        elif self.normalization == 'global_unity_xy':
                            if (self.max_intensity[0] - self.min_intensity[0]) != 0:
                                img = (img - self.min_intensity[0]) / (self.max_intensity[0] - self.min_intensity[0])
                            if (self.max_intensity[1] - self.min_intensity[1]) != 0:
                                anno = (img - self.min_intensity[1]) / (self.max_intensity[1] - self.min_intensity[1])
                        elif self.normalization == 'global_negpos_xy':
                            if (self.max_intensity[0] - self.min_intensity[0]) != 0:
                                img = 2 * (((img - self.min_intensity[0]) / (self.max_intensity[0] - self.min_intensity[0])) - 0.5)
                            if (self.max_intensity[1] - self.min_intensity[1]) != 0:
                                anno = 2 * (((anno - self.min_intensity[1]) / (self.max_intensity[1] - self.min_intensity[1])) - 0.5)
                        elif self.normalization == 'none':
                            pass
                        else:
                            raise ValueError('Normalization type must be either samplewise_unity_x,'
                                             + ' samplewise_negpos_x, global_unity_x, global_negpos_x,'
                                             + ' samplewise_unity_xy, samplewise_negpos_xy, global_unity_xy,'
                                             + ' global_negpos_xy, or none.')

                    if orig_shape[-1] == 1:
                        img = np.take(img, 0, axis=-1)
                        img = np.expand_dims(img, axis=-1)

                    batch_X.append(img)
                    if self.to_categorical:
                        if self.num_classes is None:
                            raise ValueError('If converting to categorical variables, you must specify the number'
                                             + ' of classes.')
                        else:
                            integers = np.isin(anno, orig_classes)
                            anno = np.multiply(anno, integers)
                            anno = to_categorical(anno, num_classes=self.num_classes)
                    batch_y.append(anno)

                f_imgs.close()
                f_annos.close()

                current_train += self.batch_size

                batch_X = np.array(batch_X)
                batch_y = np.array(batch_y)
                ret = (batch_X, batch_y)

                yield ret

            elif self.subset == 'validation':
                if self.val_dataset_size is None:
                    raise ValueError('Zero validation data was reserved. If you want to generate a validation set,'
                                      + ' set 0 < validation_split < 1')
                else:
                    batch_X, batch_y = [], []

                    if current_val >= self.val_dataset_size:
                        current_val = 0

                    batch_indices = self.val_dataset_indices[current_val:current_val + self.batch_size]
                    f_imgs = h5py.File(self.imgs_hdf5_path, 'r')
                    f_annos = h5py.File(self.annos_hdf5_path, 'r')
                    for i in batch_indices:
                        sample_name = self.validation_keys[i]
                        img = f_imgs[sample_name].value
                        if len(img.shape) == 2:
                            img = np.expand_dims(img, axis=-1)
                        anno = f_annos[sample_name].value
                        anno = anno.astype('float32')
                        if self.to_categorical:
                            orig_classes = np.unique(anno)
                        else:
                            orig_classes = None

                        if img.shape[0] != anno.shape[0] or img.shape[1] != anno.shape[1]:
                            raise ValueError('Images and annotations do not have the same number of rows and columns.')

                        if len(anno.shape) == 2:
                            anno = np.expand_dims(anno, axis=-1)

                        if self.repeat_chans:
                            img = np.repeat(img, self.chan_repititions, axis=-1)

                        if self.featurewise_center:
                            mean = np.mean(img, axis=(0, row_axis, col_axis))
                            broadcast_shape = [1, 1, 1]
                            broadcast_shape[chan_axis - 1] = img.shape[chan_axis]
                            mean = np.reshape(mean, broadcast_shape)
                            img -= mean

                        if self.featurewise_std_normalization:
                            std = np.std(img, axis=(0, row_axis, col_axis))
                            broadcast_shape = [1, 1, 1]
                            broadcast_shape[chan_axis - 1] = img.shape[chan_axis]
                            std = np.reshape(std, broadcast_shape)
                            img /= (std + 1e-6)

                        if self.samplewise_center:
                            img -= np.mean(img, keepdims=True)

                        if self.samplewise_std_normalization:
                            img /= (np.std(img, keepdims=True) + 1e-6)

                        if self.normalization is not None:
                            if self.normalization == 'samplewise_unity_x':
                                if (np.max(img) - np.min(img)) != 0:
                                    img = (img - np.min(img)) / (np.max(img) - np.min(img))
                            elif self.normalization == 'samplewise_negpos_x':
                                if (np.max(img) - np.min(img)) != 0:
                                    img = 2 * (((img - np.min(img)) / (np.max(img) - np.min(img))) - 0.5)
                            elif self.normalization == 'global_unity_x':
                                if (self.max_intensity[0] - self.min_intensity[0]) != 0:
                                    img = (img - self.min_intensity[0]) / (self.max_intensity[0] - self.min_intensity[0])
                            elif self.normalization == 'global_negpos_x':
                                if (self.max_intensity[0] - self.min_intensity[0]) != 0:
                                    img = 2 * (((img - self.min_intensity[0]) / (self.max_intensity[0] - self.min_intensity[0])) - 0.5)
                            elif self.normalization == 'samplewise_unity_xy':
                                if (np.max(img) - np.min(img)) != 0:
                                    img = (img - np.min(img)) / (np.max(img) - np.min(img))
                                if (np.max(anno) - np.min(anno)) != 0:
                                    anno = (anno - np.min(anno)) / (np.max(anno) - np.min(anno))
                            elif self.normalization == 'samplewise_negpos_xy':
                                if (np.max(img) - np.min(img)) != 0:
                                    img = 2 * (((img - np.min(img)) / (np.max(img) - np.min(img))) - 0.5)
                                if (np.max(anno) - np.min(anno)) != 0:
                                    anno = 2 * (((anno - np.min(anno)) / (np.max(anno) - np.min(anno))) - 0.5)
                            elif self.normalization == 'global_unity_xy':
                                if (self.max_intensity[0] - self.min_intensity[0]) != 0:
                                    img = (img - self.min_intensity[0]) / (self.max_intensity[0] - self.min_intensity[0])
                                if (self.max_intensity[1] - self.min_intensity[1]) != 0:
                                    anno = (img - self.min_intensity[1]) / (self.max_intensity[1] - self.min_intensity[1])
                            elif self.normalization == 'global_negpos_xy':
                                if (self.max_intensity[0] - self.min_intensity[0]) != 0:
                                    img = 2 * (((img - self.min_intensity[0]) / (self.max_intensity[0] - self.min_intensity[0])) - 0.5)
                                if (self.max_intensity[1] - self.min_intensity[1]) != 0:
                                    anno = 2 * (((anno - self.min_intensity[1]) / (self.max_intensity[1] - self.min_intensity[1])) - 0.5)
                            elif self.normalization == 'none':
                                pass
                            else:
                                raise ValueError('Normalization type must be either samplewise_unity_x,'
                                                 + ' samplewise_negpos_x, global_unity_x, global_negpos_x,'
                                                 + ' samplewise_unity_xy, samplewise_negpos_xy, global_unity_xy,'
                                                 + ' global_negpos_xy, or none.')

                        batch_X.append(img)
                        if self.to_categorical:
                            if self.num_classes is None:
                                raise ValueError('If converting to categorical variables, you must specify the number'
                                                 + ' of classes.')
                            else:
                                integers = np.isin(anno, orig_classes)
                                anno = np.multiply(anno, integers)
                                anno = to_categorical(anno, num_classes=self.num_classes)
                        batch_y.append(anno)

                    f_imgs.close()
                    f_annos.close()

                    current_val += self.batch_size

                    batch_X = np.array(batch_X)
                    batch_y = np.array(batch_y)
                    ret = (batch_X, batch_y)

                    yield ret

            elif self.subset == 'test':
                batch_X = []

                if current_test >= self.test_dataset_size:
                    current_test = 0

                batch_indices = self.test_dataset_indices[current_test:current_test + self.batch_size]
                f_imgs = h5py.File(self.imgs_hdf5_path, 'r')
                for i in batch_indices:
                    sample_name = self.test_keys[i]
                    img = f_imgs[sample_name].value
                    if len(img.shape) == 2:
                        img = np.expand_dims(img, axis=-1)

                    if self.repeat_chans:
                        img = np.repeat(img, self.chan_repititions, axis=-1)

                    if self.normalization is not None:
                        if self.normalization == 'samplewise_unity_x':
                            if (np.max(img) - np.min(img)) != 0:
                                img = (img - np.min(img)) / (np.max(img) - np.min(img))
                        elif self.normalization == 'samplewise_negpos_x':
                            if (np.max(img) - np.min(img)) != 0:
                                img = 2 * (((img - np.min(img)) / (np.max(img) - np.min(img))) - 0.5)
                        elif self.normalization == 'global_unity_x':
                            if (self.max_intensity[0] - self.min_intensity[0]) != 0:
                                img = (img - self.min_intensity[0]) / (self.max_intensity[0] - self.min_intensity[0])
                        elif self.normalization == 'global_negpos_x':
                            if (self.max_intensity[0] - self.min_intensity[0]) != 0:
                                img = 2 * (((img - self.min_intensity[0]) / (self.max_intensity[0] - self.min_intensity[0])) - 0.5)
                        elif self.normalization == 'samplewise_unity_xy':
                            if (np.max(img) - np.min(img)) != 0:
                                img = (img - np.min(img)) / (np.max(img) - np.min(img))
                        elif self.normalization == 'samplewise_negpos_xy':
                            if (np.max(img) - np.min(img)) != 0:
                                img = 2 * (((img - np.min(img)) / (np.max(img) - np.min(img))) - 0.5)
                        elif self.normalization == 'global_unity_xy':
                            if (self.max_intensity[0] - self.min_intensity[0]) != 0:
                                img = (img - self.min_intensity[0]) / (self.max_intensity[0] - self.min_intensity[0])
                        elif self.normalization == 'global_negpos_xy':
                            if (self.max_intensity[0] - self.min_intensity[0]) != 0:
                                img = 2 * (((img - self.min_intensity[0]) / (self.max_intensity[0] - self.min_intensity[0])) - 0.5)
                        elif self.normalization == 'none':
                            pass
                        else:
                            raise ValueError('Normalization type must be either samplewise_unity_x,'
                                             + ' samplewise_negpos_x, global_unity_x, global_negpos_x,'
                                             + ' samplewise_unity_xy, samplewise_negpos_xy, global_unity_xy,'
                                             + ' global_negpos_xy, or none.')

                    batch_X.append(img)

                f_imgs.close()

                current_test += self.batch_size

                batch_X = np.array(batch_X)
                ret = (batch_X)

                yield ret


class FCN3DDatasetGenerator(object):
    def __init__(self,
                 imgs_hdf5_path,
                 annos_hdf5_path=None,
                 rotation_range=0,
                 width_shift_range=0.,
                 height_shift_range=0.,
                 shear_range=0.,
                 zoom_range=0.,
                 flip_horizontal=False,
                 flip_vertical=False,
                 featurewise_center=False,
                 featurewise_std_normalization=False,
                 samplewise_center=False,
                 samplewise_std_normalization=False,
                 zca_epsilon=None,
                 brightness_range=(0.75, 1.25),
                 channel_shift_range=0.1,
                 shuffle_data=False,
                 rounds=1,
                 fill_mode='nearest',
                 cval=0,
                 interpolation_order=1,
                 seed=None,
                 batch_size=4,
                 validation_split=0.0,
                 subset='train',
                 normalization=None,
                 min_intensity=[0.],
                 max_intensity=[0.],
                 categorical_labels=False,
                 num_classes=None,
                 repeat_chans=False,
                 chan_repititions=0,
                 apply_aug=False):
        self.imgs_hdf5_path = imgs_hdf5_path
        self.annos_hdf5_path = annos_hdf5_path
        self.batch_size = batch_size
        self.keys = get_keys(self.imgs_hdf5_path)
        self.keys.sort(key=int)
        self.shuffle_data = shuffle_data
        self.rotation_range = rotation_range
        self.width_shift_range = width_shift_range
        self.height_shift_range = height_shift_range
        self.shear_range = shear_range
        self.zoom_range = [1 - zoom_range, 1 + zoom_range]
        self.flip_horizontal = flip_horizontal
        self.flip_vertical = flip_vertical
        self.featurewise_center = featurewise_center
        self.featurewise_std_normalization = featurewise_std_normalization
        self.samplewise_center = samplewise_center
        self.samplewise_std_normalization = samplewise_std_normalization
        self.zca_epsilon = zca_epsilon
        self.brightness_range = brightness_range
        self.channel_shift_range = channel_shift_range
        self.fill_mode = fill_mode
        self.cval = cval
        self.interpolation_order = interpolation_order
        self.rounds = rounds
        self.seed = seed
        self.validation_split = validation_split
        self.subset = subset
        if self.subset == 'train' or self.subset == 'validation':
            self.proceed = check_keys(self.imgs_hdf5_path, self.annos_hdf5_path)
        else:
            self.proceed = None
        self.normalization = normalization
        self.min_intensity = min_intensity
        self.max_intensity = max_intensity
        self.to_categorical = categorical_labels
        self.num_classes = num_classes
        if self.subset == 'train':
            if self.shuffle_data:
                shuffle(self.keys)
            self.train_dataset_size = np.floor(len(self.keys) * (1 - self.validation_split))
            self.test_dataset_size = None
            self.train_dataset_indices = np.arange(self.train_dataset_size, dtype=np.int32)
            self.test_dataset_indices = None
            if self.validation_split > 0.:
                self.val_dataset_size = np.ceil(len(self.keys) * self.validation_split)
                self.val_dataset_indices = np.arange(self.val_dataset_size, dtype=np.int32)
                split_idx = int(np.ceil(len(self.keys) * self.validation_split))
                self.validation_keys = self.keys[:split_idx]
                self.train_keys = self.keys[split_idx:]
            else:
                self.val_dataset_size = None
                self.val_dataset_indices = None
                self.validation_keys = None
                self.train_keys = self.keys
            self.test_keys = None
        elif self.subset == 'validation':
            self.train_dataset_size = None
            self.val_dataset_size = len(self.keys)
            self.test_dataset_size = None
            self.train_dataset_indices = None
            self.val_dataset_indices = np.arange(self.val_dataset_size, dtype=np.int32)
            self.test_dataset_indices = None
            self.train_keys = None
            self.validation_keys = self.keys
            self.test_keys = None
        elif self.subset == 'test':
            self.train_dataset_size = None
            self.val_dataset_size = None
            self.test_dataset_size = len(self.keys)
            self.train_dataset_indices = None
            self.val_dataset_indices = None
            self.test_dataset_indices = np.arange(self.test_dataset_size, dtype=np.int32)
            self.train_keys = None
            self.validation_keys = None
            self.test_keys = self.keys
        else:
            raise ValueError('Invalid subset specified. Valid values are train, validation, or test.')
        self.repeat_chans = repeat_chans
        self.chan_repititions = chan_repititions
        self.apply_augmentation = apply_aug

    def __len__(self):
        """
        The "length" of the generator is the number of batches expected.
        :return: the expected number of batches that will be produced by this generator.
        """
        if self.subset == 'train':
            return int(np.ceil(self.rounds * self.train_dataset_size / self.batch_size))
        elif self.subset == 'validation':
            if self.val_dataset_size is None:
                raise ValueError('Zero validation data was reserved. If you want to generate a validation set,'
                                 + ' set 0 < validation_split < 1')
            else:
                return int(np.ceil(self.val_dataset_size / self.batch_size))
        elif self.subset == 'test':
            return int(np.ceil(self.test_dataset_size / self.batch_size))
        else:
            raise ValueError('Invalidation subset defined. Only train, validation, and test are valid subsets.')

    def generate(self):
        """
        Reads in data from an HDF5 file, applies augmentation chain (if
        desired), shuffles and batches the data.
        """
        row_axis = 0
        col_axis = 1
        slice_axis = 2
        chan_axis = 3
        if self.proceed is False:
            raise ValueError('Datset names in the X (image) and y (annotation) HDF5 files are not identical.'
                             + ' Images and annotations must be paired and have identical names in the HDF5 files.')

        if self.seed is not None:
            np.random.seed(self.seed)

        current_train = 0
        current_val = 0
        current_test = 0

        while True:
            if self.subset == 'train':
                batch_X, batch_y = [], []

                if current_train >= self.train_dataset_size:
                    current_train = 0

                    if self.shuffle_data:
                        shuffle(self.train_keys)

                batch_indices = self.train_dataset_indices[current_train:current_train + self.batch_size]
                f_imgs = h5py.File(self.imgs_hdf5_path, 'r')
                f_annos = h5py.File(self.annos_hdf5_path, 'r')

                for i in batch_indices:
                    sample_name = self.train_keys[i]
                    img = f_imgs[sample_name].value
                    if len(img.shape) == 3:
                        img = np.expand_dims(img, axis=-1)
                    anno = f_annos[sample_name].value
                    anno = anno.astype('float32')
                    if self.to_categorical:
                        orig_classes = np.unique(anno)
                    else:
                        orig_classes = None
                    orig_shape = img.shape

                    if img.shape[0] != anno.shape[0] or img.shape[1] != anno.shape[1] or img.shape[2] != anno.shape[2]:
                        raise ValueError('Images and annotations do not have the same number of rows, columns, and slices.')

                    if len(anno.shape) == 3:
                        anno = np.expand_dims(anno, axis=-1)

                    if self.repeat_chans:
                        img = np.repeat(img, self.chan_repititions, axis=-1)

                    if self.featurewise_center:
                        mean = np.mean(img, axis=(0, row_axis, col_axis, slice_axis))
                        broadcast_shape = [1, 1, 1, 1]
                        broadcast_shape[chan_axis - 1] = img.shape[chan_axis]
                        mean = np.reshape(mean, broadcast_shape)
                        img -= mean

                    if self.featurewise_std_normalization:
                        std = np.std(img, axis=(0, row_axis, col_axis, slice_axis))
                        broadcast_shape = [1, 1, 1, 1]
                        broadcast_shape[chan_axis - 1] = img.shape[chan_axis]
                        std = np.reshape(std, broadcast_shape)
                        img /= (std + 1e-6)

                    if self.samplewise_center:
                        img -= np.mean(img, keepdims=True)

                    if self.samplewise_std_normalization:
                        img /= (np.std(img, keepdims=True) + 1e-6)

                    if self.zca_epsilon is not None and self.apply_augmentation is True:
                        flat_x = np.reshape(
                            img, (img.shape[0], np.prod(img.shape[1:])))
                        sigma = np.dot(flat_x.T, flat_x) / flat_x.shape[0]
                        u, s, _ = linalg.svd(sigma)
                        s_inv = 1. / np.sqrt(s[np.newaxis] + self.zca_epsilon)
                        principal_components = (u * s_inv).dot(u.T)
                        flatx = np.reshape(img, (-1, np.prod(img.shape[1:])))
                        whitex = np.dot(flatx, principal_components)
                        img = np.reshape(whitex, img.shape)

                    if self.rotation_range:
                        theta = np.random.uniform(-self.rotation_range, self.rotation_range)
                    else:
                        theta = 0

                    if self.height_shift_range:
                        try:  # 1-D array-like or int
                            tx = np.random.choice(self.height_shift_range)
                            tx *= np.random.choice([-1, 1])
                        except ValueError:  # floating point
                            tx = np.random.uniform(-self.height_shift_range,
                                                   self.height_shift_range)
                        if np.max(self.height_shift_range) < 1:
                            tx *= img.shape[row_axis]
                    else:
                        tx = 0

                    if self.width_shift_range:
                        try:  # 1-D array-like or int
                            ty = np.random.choice(self.width_shift_range)
                            ty *= np.random.choice([-1, 1])
                        except ValueError:  # floating point
                            ty = np.random.uniform(-self.width_shift_range,
                                                   self.width_shift_range)
                        if np.max(self.width_shift_range) < 1:
                            ty *= img.shape[col_axis]
                    else:
                        ty = 0

                    if self.shear_range:
                        shear = np.random.uniform(
                            -self.shear_range,
                            self.shear_range)
                    else:
                        shear = 0

                    if self.zoom_range[0] == 1 and self.zoom_range[1] == 1:
                        zx, zy = 1, 1
                    else:
                        zx, zy = np.random.uniform(
                            self.zoom_range[0],
                            self.zoom_range[1],
                            2)

                    brightness = None
                    if self.brightness_range is not None:
                        if orig_shape[-1] == 1:
                            img = np.repeat(img, 3, axis=-1)
                        brightness = np.random.uniform(self.brightness_range[0],
                                                       self.brightness_range[1])

                    channel_shift_intensity = None
                    if self.channel_shift_range != 0:
                        channel_shift_intensity = np.random.uniform(-self.channel_shift_range,
                                                                    self.channel_shift_range)

                    flip_horizontal = (np.random.random() < 0.5) * self.flip_horizontal
                    flip_vertical = (np.random.random() < 0.5) * self.flip_vertical

                    transform_parameters_img = {'theta': theta,
                                                'tx': tx,
                                                'ty': ty,
                                                'shear': shear,
                                                'zx': zx,
                                                'zy': zy,
                                                'flip_horizontal': flip_horizontal,
                                                'flip_vertical': flip_vertical,
                                                'fill_mode': self.fill_mode,
                                                'cval': self.cval,
                                                'interpolation_order': self.interpolation_order,
                                                'brightness': brightness,
                                                'channel_shift_intensity': channel_shift_intensity}
                    transform_parameters_anno = {'theta': theta,
                                                 'tx': tx,
                                                 'ty': ty,
                                                 'shear': shear,
                                                 'zx': zx,
                                                 'zy': zy,
                                                 'flip_horizontal': flip_horizontal,
                                                 'flip_vertical': flip_vertical,
                                                 'fill_mode': self.fill_mode,
                                                 'cval': self.cval,
                                                 'interpolation_order': self.interpolation_order}

                    if self.apply_augmentation is True:
                        new_img = []
                        new_anno = []
                        for i in range(img.shape[2]):
                            aug_img = img[:, :, i, :]
                            aug_anno = anno[:, :, i, :]
                            aug_img = apply_transform(aug_img, transform_parameters_img)
                            aug_anno = apply_transform(aug_anno, transform_parameters_anno)
                            new_img.append(aug_img)
                            new_anno.append(aug_anno)
                        img = np.array(new_img)
                        img = np.transpose(img, (1, 2, 0, 3))
                        anno = np.array(new_anno)
                        anno = np.transpose(anno, (1, 2, 0, 3))

                    if self.normalization is not None:
                        if self.normalization == 'samplewise_unity_x':
                            if (np.max(img) - np.min(img)) != 0:
                                img = (img - np.min(img)) / (np.max(img) - np.min(img))
                        elif self.normalization == 'samplewise_negpos_x':
                            if (np.max(img) - np.min(img)) != 0:
                                img = 2 * (((img - np.min(img)) / (np.max(img) - np.min(img))) - 0.5)
                        elif self.normalization == 'global_unity_x':
                            if (self.max_intensity[0] - self.min_intensity[0]) != 0:
                                img = (img - self.min_intensity[0]) / (self.max_intensity[0] - self.min_intensity[0])
                        elif self.normalization == 'global_negpos_x':
                            if (self.max_intensity[0] - self.min_intensity[0]) != 0:
                                img = 2 * (((img - self.min_intensity[0]) / (self.max_intensity[0] - self.min_intensity[0])) - 0.5)
                        elif self.normalization == 'samplewise_unity_xy':
                            if (np.max(img) - np.min(img)) != 0:
                                img = (img - np.min(img)) / (np.max(img) - np.min(img))
                            if (np.max(anno) - np.min(anno)) != 0:
                                anno = (anno - np.min(anno)) / (np.max(anno) - np.min(anno))
                        elif self.normalization == 'samplewise_negpos_xy':
                            if (np.max(img) - np.min(img)) != 0:
                                img = 2 * (((img - np.min(img)) / (np.max(img) - np.min(img))) - 0.5)
                            if (np.max(anno) - np.min(anno)) != 0:
                                anno = 2 * (((anno - np.min(anno)) / (np.max(anno) - np.min(anno))) - 0.5)
                        elif self.normalization == 'global_unity_xy':
                            if (self.max_intensity[0] - self.min_intensity[0]) != 0:
                                img = (img - self.min_intensity[0]) / (self.max_intensity[0] - self.min_intensity[0])
                            if (self.max_intensity[1] - self.min_intensity[1]) != 0:
                                anno = (img - self.min_intensity[1]) / (self.max_intensity[1] - self.min_intensity[1])
                        elif self.normalization == 'global_negpos_xy':
                            if (self.max_intensity[0] - self.min_intensity[0]) != 0:
                                img = 2 * (((img - self.min_intensity[0]) / (self.max_intensity[0] - self.min_intensity[0])) - 0.5)
                            if (self.max_intensity[1] - self.min_intensity[1]) != 0:
                                anno = 2 * (((anno - self.min_intensity[1]) / (self.max_intensity[1] - self.min_intensity[1])) - 0.5)
                        elif self.normalization == 'none':
                            pass
                        else:
                            raise ValueError('Normalization type must be either samplewise_unity_x,'
                                             + ' samplewise_negpos_x, global_unity_x, global_negpos_x,'
                                             + ' samplewise_unity_xy, samplewise_negpos_xy, global_unity_xy,'
                                             + ' global_negpos_xy, or none.')

                    if orig_shape[-1] == 1:
                        img = np.take(img, 0, axis=-1)
                        img = np.expand_dims(img, axis=-1)

                    batch_X.append(img)
                    if self.to_categorical:
                        if self.num_classes is None:
                            raise ValueError('If converting to categorical variables, you must specify the number'
                                             + ' of classes.')
                        else:
                            integers = np.isin(anno, orig_classes)
                            anno = np.multiply(anno, integers)
                            anno = to_categorical(anno, num_classes=self.num_classes)
                    batch_y.append(anno)

                f_imgs.close()
                f_annos.close()

                current_train += self.batch_size

                batch_X = np.array(batch_X)
                batch_y = np.array(batch_y)
                ret = (batch_X, batch_y)

                yield ret

            elif self.subset == 'validation':
                if self.val_dataset_size is None:
                    raise ValueError('Zero validation data was reserved. If you want to generate a validation set,'
                                      + ' set 0 < validation_split < 1')
                else:
                    batch_X, batch_y = [], []

                    if current_val >= self.val_dataset_size:
                        current_val = 0

                    batch_indices = self.val_dataset_indices[current_val:current_val + self.batch_size]
                    f_imgs = h5py.File(self.imgs_hdf5_path, 'r')
                    f_annos = h5py.File(self.annos_hdf5_path, 'r')
                    for i in batch_indices:
                        sample_name = self.validation_keys[i]
                        img = f_imgs[sample_name].value
                        if len(img.shape) == 3:
                            img = np.expand_dims(img, axis=-1)
                        anno = f_annos[sample_name].value
                        anno = anno.astype('float32')
                        if self.to_categorical:
                            orig_classes = np.unique(anno)
                        else:
                            orig_classes = None

                        if img.shape[0] != anno.shape[0] or img.shape[1] != anno.shape[1] or img.shape[2] != anno.shape[2]:
                            raise ValueError('Images and annotations do not have the same number of rows, columns, and slices.')

                        if len(anno.shape) == 3:
                            anno = np.expand_dims(anno, axis=-1)

                        if self.repeat_chans:
                            img = np.repeat(img, self.chan_repititions, axis=-1)

                        if self.featurewise_center:
                            mean = np.mean(img, axis=(0, row_axis, col_axis, slice_axis))
                            broadcast_shape = [1, 1, 1, 1]
                            broadcast_shape[chan_axis - 1] = img.shape[chan_axis]
                            mean = np.reshape(mean, broadcast_shape)
                            img -= mean

                        if self.featurewise_std_normalization:
                            std = np.std(img, axis=(0, row_axis, col_axis, slice_axis))
                            broadcast_shape = [1, 1, 1, 1]
                            broadcast_shape[chan_axis - 1] = img.shape[chan_axis]
                            std = np.reshape(std, broadcast_shape)
                            img /= (std + 1e-6)

                        if self.samplewise_center:
                            img -= np.mean(img, keepdims=True)

                        if self.samplewise_std_normalization:
                            img /= (np.std(img, keepdims=True) + 1e-6)

                        if self.normalization is not None:
                            if self.normalization == 'samplewise_unity_x':
                                if (np.max(img) - np.min(img)) != 0:
                                    img = (img - np.min(img)) / (np.max(img) - np.min(img))
                            elif self.normalization == 'samplewise_negpos_x':
                                if (np.max(img) - np.min(img)) != 0:
                                    img = 2 * (((img - np.min(img)) / (np.max(img) - np.min(img))) - 0.5)
                            elif self.normalization == 'global_unity_x':
                                if (self.max_intensity[0] - self.min_intensity[0]) != 0:
                                    img = (img - self.min_intensity[0]) / (self.max_intensity[0] - self.min_intensity[0])
                            elif self.normalization == 'global_negpos_x':
                                if (self.max_intensity[0] - self.min_intensity[0]) != 0:
                                    img = 2 * (((img - self.min_intensity[0]) / (self.max_intensity[0] - self.min_intensity[0])) - 0.5)
                            elif self.normalization == 'samplewise_unity_xy':
                                if (np.max(img) - np.min(img)) != 0:
                                    img = (img - np.min(img)) / (np.max(img) - np.min(img))
                                if (np.max(anno) - np.min(anno)) != 0:
                                    anno = (anno - np.min(anno)) / (np.max(anno) - np.min(anno))
                            elif self.normalization == 'samplewise_negpos_xy':
                                if (np.max(img) - np.min(img)) != 0:
                                    img = 2 * (((img - np.min(img)) / (np.max(img) - np.min(img))) - 0.5)
                                if (np.max(anno) - np.min(anno)) != 0:
                                    anno = 2 * (((anno - np.min(anno)) / (np.max(anno) - np.min(anno))) - 0.5)
                            elif self.normalization == 'global_unity_xy':
                                if (self.max_intensity[0] - self.min_intensity[0]) != 0:
                                    img = (img - self.min_intensity[0]) / (self.max_intensity[0] - self.min_intensity[0])
                                if (self.max_intensity[1] - self.min_intensity[1]) != 0:
                                    anno = (img - self.min_intensity[1]) / (self.max_intensity[1] - self.min_intensity[1])
                            elif self.normalization == 'global_negpos_xy':
                                if (self.max_intensity[0] - self.min_intensity[0]) != 0:
                                    img = 2 * (((img - self.min_intensity[0]) / (self.max_intensity[0] - self.min_intensity[0])) - 0.5)
                                if (self.max_intensity[1] - self.min_intensity[1]) != 0:
                                    anno = 2 * (((anno - self.min_intensity[1]) / (self.max_intensity[1] - self.min_intensity[1])) - 0.5)
                            elif self.normalization == 'none':
                                pass
                            else:
                                raise ValueError('Normalization type must be either samplewise_unity_x,'
                                                 + ' samplewise_negpos_x, global_unity_x, global_negpos_x,'
                                                 + ' samplewise_unity_xy, samplewise_negpos_xy, global_unity_xy,'
                                                 + ' global_negpos_xy, or none.')

                        batch_X.append(img)
                        if self.to_categorical:
                            if self.num_classes is None:
                                raise ValueError('If converting to categorical variables, you must specify the number'
                                                 + ' of classes.')
                            else:
                                integers = np.isin(anno, orig_classes)
                                anno = np.multiply(anno, integers)
                                anno = to_categorical(anno, num_classes=self.num_classes)
                        batch_y.append(anno)

                    f_imgs.close()
                    f_annos.close()

                    current_val += self.batch_size

                    batch_X = np.array(batch_X)
                    batch_y = np.array(batch_y)
                    ret = (batch_X, batch_y)

                    yield ret

            elif self.subset == 'test':
                batch_X = []

                if current_test >= self.test_dataset_size:
                    current_test = 0

                batch_indices = self.test_dataset_indices[current_test:current_test + self.batch_size]
                f_imgs = h5py.File(self.imgs_hdf5_path, 'r')
                for i in batch_indices:
                    sample_name = self.test_keys[i]
                    img = f_imgs[sample_name].value
                    if len(img.shape) == 3:
                        img = np.expand_dims(img, axis=-1)

                    if self.repeat_chans:
                        img = np.repeat(img, self.chan_repititions, axis=-1)

                    if self.normalization is not None:
                        if self.normalization == 'samplewise_unity_x':
                            if (np.max(img) - np.min(img)) != 0:
                                img = (img - np.min(img)) / (np.max(img) - np.min(img))
                        elif self.normalization == 'samplewise_negpos_x':
                            if (np.max(img) - np.min(img)) != 0:
                                img = 2 * (((img - np.min(img)) / (np.max(img) - np.min(img))) - 0.5)
                        elif self.normalization == 'global_unity_x':
                            if (self.max_intensity[0] - self.min_intensity[0]) != 0:
                                img = (img - self.min_intensity[0]) / (self.max_intensity[0] - self.min_intensity[0])
                        elif self.normalization == 'global_negpos_x':
                            if (self.max_intensity[0] - self.min_intensity[0]) != 0:
                                img = 2 * (((img - self.min_intensity[0]) / (self.max_intensity[0] - self.min_intensity[0])) - 0.5)
                        elif self.normalization == 'samplewise_unity_xy':
                            if (np.max(img) - np.min(img)) != 0:
                                img = (img - np.min(img)) / (np.max(img) - np.min(img))
                        elif self.normalization == 'samplewise_negpos_xy':
                            if (np.max(img) - np.min(img)) != 0:
                                img = 2 * (((img - np.min(img)) / (np.max(img) - np.min(img))) - 0.5)
                        elif self.normalization == 'global_unity_xy':
                            if (self.max_intensity[0] - self.min_intensity[0]) != 0:
                                img = (img - self.min_intensity[0]) / (self.max_intensity[0] - self.min_intensity[0])
                        elif self.normalization == 'global_negpos_xy':
                            if (self.max_intensity[0] - self.min_intensity[0]) != 0:
                                img = 2 * (((img - self.min_intensity[0]) / (self.max_intensity[0] - self.min_intensity[0])) - 0.5)
                        elif self.normalization == 'none':
                            pass
                        else:
                            raise ValueError('Normalization type must be either samplewise_unity_x,'
                                             + ' samplewise_negpos_x, global_unity_x, global_negpos_x,'
                                             + ' samplewise_unity_xy, samplewise_negpos_xy, global_unity_xy,'
                                             + ' global_negpos_xy, or none.')

                    batch_X.append(img)

                f_imgs.close()

                current_test += self.batch_size

                batch_X = np.array(batch_X)
                ret = (batch_X)

                yield ret


class SSD2DDatasetGenerator(object):
    def __init__(self,
                 imgs_hdf5_path,
                 annos_hdf5_path=None,
                 shuffle_data=False,
                 rounds=1,
                 seed=None,
                 batch_size=4,
                 validation_split=0.0,
                 subset='train',
                 normalization=None,
                 min_intensity=[0.],
                 max_intensity=[0.],
                 repeat_chans=False,
                 chan_repititions=0):
        labels_output_format = ('class_id', 'xmin', 'ymin', 'xmax', 'ymax')
        self.labels_format = {'class_id': labels_output_format.index('class_id'),
                              'xmin': labels_output_format.index('xmin'),
                              'ymin': labels_output_format.index('ymin'),
                              'xmax': labels_output_format.index('xmax'),
                              'ymax': labels_output_format.index('ymax')}
        self.imgs_hdf5_path = imgs_hdf5_path
        self.annos_hdf5_path = annos_hdf5_path
        self.batch_size = batch_size
        self.keys = get_keys(self.imgs_hdf5_path)
        self.keys.sort(key=int)
        self.shuffle_data = shuffle_data
        self.rounds = rounds
        self.seed = seed
        self.validation_split = validation_split
        self.subset = subset
        if self.subset == 'train' or self.subset == 'validation':
            self.proceed = check_keys(self.imgs_hdf5_path, self.annos_hdf5_path)
        else:
            self.proceed = None
        self.normalization = normalization
        self.min_intensity = min_intensity
        self.max_intensity = max_intensity
        if self.subset == 'train':
            if self.shuffle_data:
                shuffle(self.keys)
            self.train_dataset_size = np.floor(len(self.keys) * (1 - self.validation_split))
            self.test_dataset_size = None
            self.train_dataset_indices = np.arange(self.train_dataset_size, dtype=np.int32)
            self.test_dataset_indices = None
            if self.validation_split > 0.:
                self.val_dataset_size = np.ceil(len(self.keys) * self.validation_split)
                self.val_dataset_indices = np.arange(self.val_dataset_size, dtype=np.int32)
                split_idx = int(np.ceil(len(self.keys) * self.validation_split))
                self.validation_keys = self.keys[:split_idx]
                self.train_keys = self.keys[split_idx:]
            else:
                self.val_dataset_size = None
                self.val_dataset_indices = None
                self.validation_keys = None
                self.train_keys = self.keys
            self.test_keys = None
        elif self.subset == 'validation':
            self.train_dataset_size = None
            self.val_dataset_size = len(self.keys)
            self.test_dataset_size = None
            self.train_dataset_indices = None
            self.val_dataset_indices = np.arange(self.val_dataset_size, dtype=np.int32)
            self.test_dataset_indices = None
            self.train_keys = None
            self.validation_keys = self.keys
            self.test_keys = None
        elif self.subset == 'test':
            self.train_dataset_size = None
            self.val_dataset_size = None
            self.test_dataset_size = len(self.keys)
            self.train_dataset_indices = None
            self.val_dataset_indices = None
            self.test_dataset_indices = np.arange(self.test_dataset_size, dtype=np.int32)
            self.train_keys = None
            self.validation_keys = None
            self.test_keys = self.keys
        else:
            raise ValueError('Invalid subset specified. Valid values are train, validation, or test.')
        self.repeat_chans = repeat_chans
        self.chan_repititions = chan_repititions

    def __len__(self):
        """
        The "length" of the generator is the number of batches expected.
        :return: the expected number of batches that will be produced by this generator.
        """
        if self.subset == 'train':
            return int(np.ceil(self.rounds * self.train_dataset_size / self.batch_size))
        elif self.subset == 'validation':
            if self.val_dataset_size is None:
                raise ValueError('Zero validation data was reserved. If you want to generate a validation set,'
                                 + ' set 0 < validation_split < 1')
            else:
                return int(np.ceil(self.val_dataset_size / self.batch_size))
        elif self.subset == 'test':
            return int(np.ceil(self.test_dataset_size / self.batch_size))
        else:
            raise ValueError('Invalidation subset defined. Only train, validation, and test are valid subsets.')

    def generate(self, transformations=[], label_encoder=None):
        """
        Reads in data from an HDF5 file, applies augmentation chain (if
        desired), shuffles and batches the data.
        """
        if self.proceed is False:
            raise ValueError('Datset names in the X (image) and y (annotation) HDF5 files are not identical.'
                             + ' Images and annotations must be paired and have identical names in the HDF5 files.')

        if self.seed is not None:
            np.random.seed(self.seed)

        current_train = 0
        current_val = 0
        current_test = 0

        box_filter = BoxFilter(check_overlap=False,
                               check_min_area=False,
                               check_degenerate=True,
                               labels_format=self.labels_format)

        for transform in transformations:
            transform.labels_format = self.labels_format

        while True:
            if self.subset == 'train':
                batch_X, batch_y = [], []

                if current_train >= self.train_dataset_size:
                    current_train = 0

                    if self.shuffle_data:
                        shuffle(self.train_keys)

                batch_indices = self.train_dataset_indices[current_train:current_train + self.batch_size]
                f_imgs = h5py.File(self.imgs_hdf5_path, 'r')
                f_annos = h5py.File(self.annos_hdf5_path, 'r')
                for i in batch_indices:
                    cont = True
                    sample_name = self.train_keys[i]
                    img = f_imgs[sample_name].value
                    if len(img.shape) == 2:
                        img = np.expand_dims(img, axis=-1)
                    anno = f_annos[sample_name].value
                    anno = anno.astype('float32')
                    if len(anno.shape) == 1:
                        anno = np.expand_dims(anno, axis=0)

                    if anno.shape[1] != 5:
                        raise ValueError('Annotation format must be: class, xmin, ymin, xmax, ymax.')

                    if self.repeat_chans:
                        img = np.repeat(img, self.chan_repititions, axis=-1)

                    if self.normalization is not None:
                        if self.normalization == 'samplewise_unity_x':
                            if (np.max(img) - np.min(img)) != 0:
                                img = (img - np.min(img)) / (np.max(img) - np.min(img))
                        elif self.normalization == 'samplewise_negpos_x':
                            if (np.max(img) - np.min(img)) != 0:
                                img = 2 * (((img - np.min(img)) / (np.max(img) - np.min(img))) - 0.5)
                        elif self.normalization == 'global_unity_x':
                            if (self.max_intensity[0] - self.min_intensity[0]) != 0:
                                img = (img - self.min_intensity[0]) / (self.max_intensity[0] - self.min_intensity[0])
                        elif self.normalization == 'global_negpos_x':
                            if (self.max_intensity[0] - self.min_intensity[0]) != 0:
                                img = 2 * (((img - self.min_intensity[0]) / (self.max_intensity[0] - self.min_intensity[0])) - 0.5)
                        elif self.normalization == 'samplewise_unity_xy':
                            if (np.max(img) - np.min(img)) != 0:
                                img = (img - np.min(img)) / (np.max(img) - np.min(img))
                        elif self.normalization == 'samplewise_negpos_xy':
                            if (np.max(img) - np.min(img)) != 0:
                                img = 2 * (((img - np.min(img)) / (np.max(img) - np.min(img))) - 0.5)
                        elif self.normalization == 'global_unity_xy':
                            if (self.max_intensity[0] - self.min_intensity[0]) != 0:
                                img = (img - self.min_intensity[0]) / (self.max_intensity[0] - self.min_intensity[0])
                        elif self.normalization == 'global_negpos_xy':
                            if (self.max_intensity[0] - self.min_intensity[0]) != 0:
                                img = 2 * (((img - self.min_intensity[0]) / (self.max_intensity[0] - self.min_intensity[0])) - 0.5)
                        elif self.normalization == 'none':
                            pass
                        else:
                            raise ValueError('Normalization type must be either samplewise_unity_x,'
                                             + ' samplewise_negpos_x, global_unity_x, global_negpos_x,'
                                             + ' samplewise_unity_xy, samplewise_negpos_xy, global_unity_xy,'
                                             + ' global_negpos_xy, or none.')

                        if anno.size == 0:
                            cont = False
                            pass

                        if transformations:
                            for transform in transformations:
                                img, anno = transform(img, anno)

                                if img is None:
                                    cont = False
                                    pass

                        xmin = self.labels_format['xmin']
                        ymin = self.labels_format['ymin']
                        xmax = self.labels_format['xmax']
                        ymax = self.labels_format['ymax']

                        if np.any(anno[:, xmax] - anno[:, xmin] <= 0) or np.any(anno[:, ymax] - anno[:, ymin] <= 0):
                            anno = box_filter(anno)
                            if anno.size == 0:
                                cont = False
                                pass

                    if cont is True:
                        if len(img.shape) == 2:
                            img = np.expand_dims(img, axis=-1)
                        batch_X.append(img)
                        batch_y.append(anno)

                f_imgs.close()
                f_annos.close()

                current_train += self.batch_size

                ret = []
                ret.append(np.array(batch_X))
                ret.append(label_encoder(batch_y, diagnostics=False))

                yield ret

            elif self.subset == 'validation':
                if self.val_dataset_size is None:
                    raise ValueError('Zero validation data was reserved. If you want to generate a validation set,'
                                      + ' set 0 < validation_split < 1')
                else:
                    batch_X, batch_y = [], []

                    if current_val >= self.val_dataset_size:
                        current_val = 0

                    batch_indices = self.val_dataset_indices[current_val:current_val + self.batch_size]
                    f_imgs = h5py.File(self.imgs_hdf5_path, 'r')
                    f_annos = h5py.File(self.annos_hdf5_path, 'r')
                    for i in batch_indices:
                        cont = True
                        sample_name = self.validation_keys[i]
                        img = f_imgs[sample_name].value
                        if len(img.shape) == 2:
                            img = np.expand_dims(img, axis=-1)
                        anno = f_annos[sample_name].value
                        anno = anno.astype('float32')
                        if len(anno.shape) == 1:
                            anno = np.expand_dims(anno, axis=0)

                        if anno.shape[1] != 5:
                            raise ValueError('Annotation format must be: class, xmin, ymin, xmax, ymax.')

                        if self.repeat_chans:
                            img = np.repeat(img, self.chan_repititions, axis=-1)

                        if self.normalization is not None:
                            if self.normalization == 'samplewise_unity_x':
                                if (np.max(img) - np.min(img)) != 0:
                                    img = (img - np.min(img)) / (np.max(img) - np.min(img))
                            elif self.normalization == 'samplewise_negpos_x':
                                if (np.max(img) - np.min(img)) != 0:
                                    img = 2 * (((img - np.min(img)) / (np.max(img) - np.min(img))) - 0.5)
                            elif self.normalization == 'global_unity_x':
                                if (self.max_intensity[0] - self.min_intensity[0]) != 0:
                                    img = (img - self.min_intensity[0]) / (self.max_intensity[0] - self.min_intensity[0])
                            elif self.normalization == 'global_negpos_x':
                                if (self.max_intensity[0] - self.min_intensity[0]) != 0:
                                    img = 2 * (((img - self.min_intensity[0]) / (self.max_intensity[0] - self.min_intensity[0])) - 0.5)
                            elif self.normalization == 'samplewise_unity_xy':
                                if (np.max(img) - np.min(img)) != 0:
                                    img = (img - np.min(img)) / (np.max(img) - np.min(img))
                            elif self.normalization == 'samplewise_negpos_xy':
                                if (np.max(img) - np.min(img)) != 0:
                                    img = 2 * (((img - np.min(img)) / (np.max(img) - np.min(img))) - 0.5)
                            elif self.normalization == 'global_unity_xy':
                                if (self.max_intensity[0] - self.min_intensity[0]) != 0:
                                    img = (img - self.min_intensity[0]) / (self.max_intensity[0] - self.min_intensity[0])
                            elif self.normalization == 'global_negpos_xy':
                                if (self.max_intensity[0] - self.min_intensity[0]) != 0:
                                    img = 2 * (((img - self.min_intensity[0]) / (self.max_intensity[0] - self.min_intensity[0])) - 0.5)
                            elif self.normalization == 'none':
                                pass
                            else:
                                raise ValueError('Normalization type must be either samplewise_unity_x,'
                                                 + ' samplewise_negpos_x, global_unity_x, global_negpos_x,'
                                                 + ' samplewise_unity_xy, samplewise_negpos_xy, global_unity_xy,'
                                                 + ' global_negpos_xy, or none.')

                        xmin = self.labels_format['xmin']
                        ymin = self.labels_format['ymin']
                        xmax = self.labels_format['xmax']
                        ymax = self.labels_format['ymax']

                        if np.any(anno[:, xmax] - anno[:, xmin] <= 0) or np.any(anno[:, ymax] - anno[:, ymin] <= 0):
                            anno = box_filter(anno)
                            if anno.size == 0:
                                cont = False
                                pass

                        if cont is True:
                            batch_X.append(img)
                            batch_y.append(anno)

                    f_imgs.close()
                    f_annos.close()

                    current_val += self.batch_size

                    ret = []
                    ret.append(np.array(batch_X))
                    ret.append(label_encoder(batch_y, diagnostics=False))

                    yield ret

            elif self.subset == 'test':
                batch_X = []

                if current_test >= self.test_dataset_size:
                    current_test = 0

                batch_indices = self.test_dataset_indices[current_test:current_test + self.batch_size]
                f_imgs = h5py.File(self.imgs_hdf5_path, 'r')
                for i in batch_indices:
                    sample_name = self.test_keys[i]
                    img = f_imgs[sample_name].value
                    if len(img.shape) == 2:
                        img = np.expand_dims(img, axis=-1)

                    if self.repeat_chans:
                        img = np.repeat(img, self.chan_repititions, axis=-1)

                    if self.normalization is not None:
                        if self.normalization == 'samplewise_unity_x':
                            if (np.max(img) - np.min(img)) != 0:
                                img = (img - np.min(img)) / (np.max(img) - np.min(img))
                        elif self.normalization == 'samplewise_negpos_x':
                            if (np.max(img) - np.min(img)) != 0:
                                img = 2 * (((img - np.min(img)) / (np.max(img) - np.min(img))) - 0.5)
                        elif self.normalization == 'global_unity_x':
                            if (self.max_intensity[0] - self.min_intensity[0]) != 0:
                                img = (img - self.min_intensity[0]) / (self.max_intensity[0] - self.min_intensity[0])
                        elif self.normalization == 'global_negpos_x':
                            if (self.max_intensity[0] - self.min_intensity[0]) != 0:
                                img = 2 * (((img - self.min_intensity[0]) / (self.max_intensity[0] - self.min_intensity[0])) - 0.5)
                        elif self.normalization == 'samplewise_unity_xy':
                            if (np.max(img) - np.min(img)) != 0:
                                img = (img - np.min(img)) / (np.max(img) - np.min(img))
                        elif self.normalization == 'samplewise_negpos_xy':
                            if (np.max(img) - np.min(img)) != 0:
                                img = 2 * (((img - np.min(img)) / (np.max(img) - np.min(img))) - 0.5)
                        elif self.normalization == 'global_unity_xy':
                            if (self.max_intensity[0] - self.min_intensity[0]) != 0:
                                img = (img - self.min_intensity[0]) / (self.max_intensity[0] - self.min_intensity[0])
                        elif self.normalization == 'global_negpos_xy':
                            if (self.max_intensity[0] - self.min_intensity[0]) != 0:
                                img = 2 * (((img - self.min_intensity[0]) / (self.max_intensity[0] - self.min_intensity[0])) - 0.5)
                        elif self.normalization == 'none':
                            pass
                        else:
                            raise ValueError('Normalization type must be either samplewise_unity_x,'
                                             + ' samplewise_negpos_x, global_unity_x, global_negpos_x,'
                                             + ' samplewise_unity_xy, samplewise_negpos_xy, global_unity_xy,'
                                             + ' global_negpos_xy, or none.')

                    batch_X.append(img)

                f_imgs.close()

                current_test += self.batch_size

                batch_X = np.array(batch_X)
                ret = (batch_X)

                yield ret
