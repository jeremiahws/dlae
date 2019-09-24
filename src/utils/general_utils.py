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
"""src/utils/general_utils.py"""


import h5py
import json


def str2bool(string):
    if string == 'True':
        return True

    elif string == 'False':
        return False

    else:
        raise ValueError


def load_config(config_file):
    with open(config_file, 'r') as f:
        configs = json.load(f)

    f.close()

    return configs


def read_hdf5(file):
    f = h5py.File(file, 'r')
    key = list(f.keys())[0]
    data = f[key].value
    f.close()

    return data


def read_hdf5_multientry(file):
    f = h5py.File(file, 'r')
    keys = list(f.keys())
    data = []
    for key in keys:
        data.append(f[key].value)
    f.close()

    return data


def write_hdf5_multientry(file, data):
    f = h5py.File(file, 'w')
    for i in range(len(data)):
        f.create_dataset(str(i), data=data[i])
    f.close()

    return


def write_hdf5(file, data):
    f = h5py.File(file, 'w')
    f.create_dataset('predictions', data=data)
    f.close()

    return


def read_bbd_X_Y(X_path, Y_path):
    f_X = h5py.File(X_path, 'r')
    f_X_keys = list(f_X.keys())
    f_Y = h5py.File(Y_path, 'r')
    f_Y_keys = list(f_Y.keys())
    different = list(set(f_X_keys) - set(f_Y_keys))

    if any(different):
        print("X and Y datasets have different dataset names in the HDF5 file.")
        print("Please make each image and bounding boxes have the same names in the HDF5 files.")
        return

    else:
        images = [f_X[key].value for key in f_X_keys]
        annos = [f_Y[key].value for key in f_Y_keys]
        return images, annos


def check_keys(X_path, Y_path):
    f_X = h5py.File(X_path, 'r')
    f_X_keys = list(f_X.keys())
    f_Y = h5py.File(Y_path, 'r')
    f_Y_keys = list(f_Y.keys())
    different = list(set(f_X_keys) - set(f_Y_keys))

    if any(different):
        return False

    else:
        return True


def get_keys(path):
    f = h5py.File(path, 'r')
    f_keys = list(f.keys())
    f.close()

    return f_keys