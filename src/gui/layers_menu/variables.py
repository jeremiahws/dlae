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
"""src/gui/layers_menu/variables.py"""


import tkinter as tk


class LayersMenuVariables(object):
    def __init__(self):
        ##############################################################################
        # Variables for convolutional layers submenu
        ##############################################################################
        self.o_padding = ('same', 'valid')
        self.s_padding = tk.StringVar(value=self.o_padding[0])
        self.o_initializer = ('he_normal', 'he_uniform', 'glorot_normal', 'glorot_uniform', 'lecun_normal', 'lecun_uniform', 'zeros', 'ones', 'random_normal', 'random_uniform', 'truncated_normal', 'orthogonal')
        self.s_initializer = tk.StringVar(value=self.o_initializer[0])
        self.o_kernel_regularizer = ('None', 'l1', 'l2', 'l1_l2')
        self.s_kernel_regularizer = tk.StringVar(value=self.o_kernel_regularizer[0])
        self.o_activity_regularizer = ('None', 'l1', 'l2', 'l1_l2')
        self.s_activity_regularizer = tk.StringVar(value=self.o_activity_regularizer[0])
        self.s_conv_layer_maps = tk.StringVar(value="64")
        self.s_conv_layer_kernel = tk.StringVar(value="(3, 3)")
        self.s_conv_layer_stride = tk.StringVar(value="(1, 1)")
        self.s_r_conv_layer_upsample = tk.StringVar(value="(2, 2)")
        self.s_dilation_rate = tk.StringVar(value="(1, 1)")
        self.s_upsample_size = tk.StringVar(value="(2, 2)")
        self.s_zeropad = tk.StringVar(value="((1, 1), (1, 1))")
        self.s_cropping = tk.StringVar(value="((0, 0), (0, 0))")
        self.s_l1 = tk.StringVar(value="0.001")
        self.s_l2 = tk.StringVar(value="0.001")

        ##############################################################################
        # Variables for pooling layers submenu
        ##############################################################################
        self.s_pool_size = tk.StringVar(value="(2, 2)")
        self.s_pool_stride = tk.StringVar(value="(2, 2)")

        ##############################################################################
        # Variables for utility layers submenu
        ##############################################################################
        self.s_input_size = tk.StringVar(value="(512, 512, 1)")
        self.s_reshape_dims = tk.StringVar(value="(262144, 1)")
        self.s_drop_rate = tk.StringVar(value="0.5")
        self.s_dense_size = tk.StringVar(value="1024")
        self.s_momentum = tk.StringVar(value="0.99")
        self.s_epsilon = tk.StringVar(value="0.001")
        self.o_activation_type = ('relu', 'tanh', 'sigmoid', 'softmax', 'hard_sigmoid', 'linear', 'softplus', 'softsign', 'selu')
        self.s_activation_type = tk.StringVar(value=self.o_activation_type[0])
        self.s_permute_dims = tk.StringVar(value="(1, 0, 2)")
        self.s_spatial_drop_rate = tk.StringVar(value="0.5")
        self.s_gauss_drop_rate = tk.StringVar(value="0.5")
        self.s_alpha_drop_rate = tk.StringVar(value="0.5")
        self.s_gauss_noise_std = tk.StringVar(value="0.1")
        self.o_skip_type = ('concatenate', 'add', 'subtract', 'multiply', 'average', 'maximum')
        self.s_skip_type = tk.StringVar(value=self.o_skip_type[0])

        ##############################################################################
        # Variables for advanced activations submenu
        ##############################################################################
        self.s_act_param = tk.StringVar(value="0.2")

        ##############################################################################
        # Variables for pretrained networks submenu
        ##############################################################################
        self.bool_include_top = tk.BooleanVar(value=False)
        self.o_weights_to_load = ('imagenet', 'none')
        self.s_weights_to_load = tk.StringVar(value=self.o_weights_to_load[0])
        self.bool_include_skips = tk.BooleanVar(value=True)
        self.bool_include_hooks = tk.BooleanVar(value=False)
