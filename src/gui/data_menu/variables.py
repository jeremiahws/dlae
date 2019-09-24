# Copyright 2019 Jeremiah Sanders.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""src/gui/data_menu/variables.py"""


import tkinter as tk


class DataMenuVariables(object):
    def __init__(self):
        '''
        Creates variables for GUI entries in the data menu.
        '''
        self.train_X_path = ''
        self.s_train_X_path = tk.StringVar(value=self.train_X_path)
        self.train_y_path = ''
        self.s_train_y_path = tk.StringVar(value=self.train_y_path)
        self.val_X_path = ''
        self.s_val_X_path = tk.StringVar(value=self.val_X_path)
        self.val_y_path = ''
        self.s_val_y_path = tk.StringVar(value=self.val_y_path)
        self.test_X_path = ''
        self.s_test_X_path = tk.StringVar(value=self.test_X_path)

        self.s_data_min = tk.StringVar(value='0.0')
        self.s_data_max = tk.StringVar(value='4096.0')
        self.o_image_context = ('2D', '3D')
        self.s_image_context = tk.StringVar(value=self.o_image_context[0])
        self.o_normalization_type = ('samplewise_unity_x', 'samplewise_negpos_x', 'global_unity_x',
                                     'global_negpos_x', 'samplewise_unity_xy', 'samplewise_negpos_xy',
                                     'global_unity_xy', 'global_negpos_xy', 'none')
        self.s_normalization_type = tk.StringVar(value=self.o_normalization_type[0])
        self.bool_to_categorical = tk.BooleanVar(value=False)
        self.s_num_categories = tk.StringVar(value='2')
        self.bool_weight_loss = tk.BooleanVar(value=False)
        self.bool_repeatX = tk.BooleanVar(value=False)
        self.s_repeatX = tk.StringVar(value='3')

        self.bool_augmentation = tk.BooleanVar(value=True)
        self.bool_fw_centering = tk.BooleanVar(value=False)
        self.bool_sw_centering = tk.BooleanVar(value=False)
        self.bool_fw_normalization = tk.BooleanVar(value=False)
        self.bool_sw_normalization = tk.BooleanVar(value=False)
        self.s_rounds = tk.StringVar(value='1')
        self.s_width_shift = tk.StringVar(value='0.0')
        self.s_height_shift = tk.StringVar(value='0.0')
        self.s_rotation_range = tk.StringVar(value='0')
        self.s_brightness_range = tk.StringVar(value='(0.8, 1.2)')
        self.s_shear_range = tk.StringVar(value='0.0')
        self.s_zoom_range = tk.StringVar(value='0.0')
        self.s_channel_shift_range = tk.StringVar(value='0.0')
        self.s_zca_epsilon = tk.StringVar(value='None')
        self.o_fill_mode = ('nearest', 'constant', 'reflect', 'wrap')
        self.s_fill_mode = tk.StringVar(value=self.o_fill_mode[0])
        self.s_cval = tk.StringVar(value='0.0')
        self.bool_horizontal_flip = tk.BooleanVar(value=True)
        self.bool_vertical_flip = tk.BooleanVar(value=True)
        self.s_random_seed = tk.StringVar(value='1')
