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
        
        Attributes:
            train_X_path --
            s_train_X_path --
            train_y_path --
            s_train_y_path --
            val_X_path --
            s_val_X_path --
            val_y_path --
            s_val_y_path --
            test_X_path --
            s_test_X_path --
            test_y_path --
            s_test_y_path --
            s_data_min --
            s_data_max --
            o_image_context --
            s_image_context --
            o_normalization_type --
            s_normalization_type --
            bool_instance_norm --
            bool_to_categorical --
            bool_weight_loss --
            bool_reshapeX --
            s_reshapeX --
            bool_permuteX --
            s_permuteX --
            bool_repeatX --
            s_repeatX --
            bool_reshapeY --
            s_reshapeY --
            bool_permuteY --
            s_permuteY --
            bool_reshape_preds --
            s_reshape_preds --
            bool_argmax_preds --
            bool_augmentation --
            bool_fw_centering --
            bool_sw_centering --
            bool_fw_normalization --
            bool_sw_normalization --
            bool_zca --
            s_zca_epsilon --
            s_width_shift --
            s_height_shift --
            s_rotation_range --
            s_brightness_range --
            s_shear_range --
            s_zoom_range --
            s_channel_shift_range --
            o_fill_mode --
            s_fill_mode --
            s_cval --
            bool_horizontal_flip --
            bool_vertical_flip --
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
        self.o_normalization_type = ('X from [0, 1]', 'X from [-1, 1]', 'X, Y from [0, 1]', 'X, Y from [-1, 1]', 'none')
        self.s_normalization_type = tk.StringVar(value=self.o_normalization_type[0])
        self.bool_to_categorical = tk.BooleanVar(value=False)
        self.bool_weight_loss = tk.BooleanVar(value=False)
        self.bool_reshapeX = tk.BooleanVar(value=False)
        self.s_reshapeX = tk.StringVar()
        self.bool_permuteX = tk.BooleanVar(value=False)
        self.s_permuteX = tk.StringVar()
        self.bool_repeatX = tk.BooleanVar(value=False)
        self.s_repeatX = tk.StringVar()
        self.bool_reshapeY = tk.BooleanVar(value=False)
        self.s_reshapeY = tk.StringVar()
        self.bool_permuteY = tk.BooleanVar(value=False)
        self.s_permuteY = tk.StringVar()

        self.bool_augmentation = tk.BooleanVar(value=True)
        self.bool_fw_centering = tk.BooleanVar(value=False)
        self.bool_sw_centering = tk.BooleanVar(value=False)
        self.bool_fw_normalization = tk.BooleanVar(value=False)
        self.bool_sw_normalization = tk.BooleanVar(value=False)
        self.s_width_shift = tk.StringVar(value='0.0')
        self.s_height_shift = tk.StringVar(value='0.0')
        self.s_rotation_range = tk.StringVar(value='0')
        self.s_brightness_range = tk.StringVar(value='(0.8, 1.2)')
        self.s_shear_range = tk.StringVar(value='0.0')
        self.s_zoom_range = tk.StringVar(value='0.0')
        self.s_channel_shift_range = tk.StringVar(value='0.0')
        self.o_fill_mode = ('nearest', 'constant', 'reflect', 'wrap')
        self.s_fill_mode = tk.StringVar(value=self.o_fill_mode[0])
        self.s_cval = tk.StringVar(value='0.0')
        self.bool_horizontal_flip = tk.BooleanVar(value=True)
        self.bool_vertical_flip = tk.BooleanVar(value=True)
