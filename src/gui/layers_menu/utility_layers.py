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
"""src/gui/layers_menu/utility_layers.py"""


import tkinter as tk


class UtilityLayers:
    def __init__(self, controller):
        self.controller = controller
        self.button_heights = 1
        self.button_widths = 15
        self.label_heights = 1
        self.label_widths = 15
        self.entry_widths = 15

        self.tl_util_layers = tk.Toplevel()
        self.tl_util_layers.title('Utility layers')
        self.tl_util_layers.resizable(width=False, height=False)
        self.tl_util_layers.wm_protocol('WM_DELETE_WINDOW', self.tl_util_layers.withdraw)
        self.b_input = tk.Button(self.tl_util_layers, text='Input', command=self.b_input_click).grid(row=0, column=0, sticky=tk.N+tk.S+tk.E+tk.W)
        self.b_reshape = tk.Button(self.tl_util_layers, text='Reshape', command=self.b_reshape_click).grid(row=1, column=0, sticky=tk.N+tk.S+tk.E+tk.W)
        self.b_dropout = tk.Button(self.tl_util_layers, text='Dropout', command=self.b_dropout_click).grid(row=2, column=0, sticky=tk.N+tk.S+tk.E+tk.W)
        self.b_dense = tk.Button(self.tl_util_layers, text='Dense', command=self.b_dense_click).grid(row=3, column=0, sticky=tk.N+tk.S+tk.E+tk.W)
        self.b_flatten = tk.Button(self.tl_util_layers, text='Flatten', command=self.b_flatten_click).grid(row=4, column=0, sticky=tk.N+tk.S+tk.E+tk.W)
        self.b_batch_norm = tk.Button(self.tl_util_layers, text='Batch Normalization', command=self.b_batch_norm_click).grid(row=5, column=0, sticky=tk.N+tk.S+tk.E+tk.W)
        self.b_activation = tk.Button(self.tl_util_layers, text='Activation', command=self.b_activation_click).grid(row=6, column=0, sticky=tk.N+tk.S+tk.E+tk.W)
        self.b_permute = tk.Button(self.tl_util_layers, text='Permute', command=self.b_permute_click).grid(row=7, column=0, sticky=tk.N+tk.S+tk.E+tk.W)
        self.b_spatial_drop_2D = tk.Button(self.tl_util_layers, text='2D Spatial Dropout', command=self.b_2D_spatial_drop).grid(row=8, column=0, sticky=tk.N+tk.S+tk.E+tk.W)
        self.b_spatial_drop_3D = tk.Button(self.tl_util_layers, text='3D Spatial Dropout', command=self.b_3D_spatial_drop).grid(row=9, column=0, sticky=tk.N+tk.S+tk.E+tk.W)
        self.b_gauss_drop = tk.Button(self.tl_util_layers, text='Gaussian Dropout', command=self.b_gauss_drop_click).grid(row=10, column=0, sticky=tk.N+tk.S+tk.E+tk.W)
        self.b_alpha_drop = tk.Button(self.tl_util_layers, text='Alpha Dropout', command=self.b_alpha_drop_click).grid(row=11, column=0, sticky=tk.N+tk.S+tk.E+tk.W)
        self.b_gauss_noise = tk.Button(self.tl_util_layers, text='Gaussian Noise', command=self.b_gauss_noise_click).grid(row=12, column=0, sticky=tk.N+tk.S+tk.E+tk.W)
        self.b_out_skip_source = tk.Button(self.tl_util_layers, text='Outer Skip Connection Source', command=self.b_out_skip_source_click).grid(row=14, column=1, sticky=tk.N+tk.S+tk.E+tk.W)
        self.b_out_skip_target = tk.Button(self.tl_util_layers, text='Outer Skip Connection Target', command=self.b_out_skip_target_click).grid(row=14, column=2, sticky=tk.N+tk.S+tk.E+tk.W)
        self.b_in_skip_source = tk.Button(self.tl_util_layers, text='Inner Skip Connection Source', command=self.b_in_skip_source_click).grid(row=15, column=1, sticky=tk.N+tk.S+tk.E+tk.W)
        self.b_in_skip_target = tk.Button(self.tl_util_layers, text='Inner Skip Connection Target', command=self.b_in_skip_target_click).grid(row=15, column=2, sticky=tk.N+tk.S+tk.E+tk.W)
        self.b_hook_source = tk.Button(self.tl_util_layers, text='Hook Connection Source', command=self.b_hook_source_click).grid(row=13, column=0, sticky=tk.N+tk.S+tk.E+tk.W)
        self.b_remove_prev = tk.Button(self.tl_util_layers, text='Remove Previous\nLayer', command=self.b_remove_prev_click).grid(row=14, column=0, rowspan=2, sticky=tk.N+tk.S+tk.E+tk.W)

        self.l_input_size = tk.Label(self.tl_util_layers, text='Input size:').grid(row=0, column=1, sticky=tk.N+tk.S+tk.E+tk.W)
        self.e_input_size = tk.Entry(self.tl_util_layers, textvariable=self.controller.layers_menu.s_input_size).grid(row=0, column=2, sticky=tk.N+tk.S+tk.E+tk.W)

        self.l_reshape_dims = tk.Label(self.tl_util_layers, text='Reshape dimensions:').grid(row=1, column=1, sticky=tk.N+tk.S+tk.E+tk.W)
        self.e_reshape_dims = tk.Entry(self.tl_util_layers, textvariable=self.controller.layers_menu.s_reshape_dims).grid(row=1, column=2, sticky=tk.N+tk.S+tk.E+tk.W)

        self.l_drop_rate = tk.Label(self.tl_util_layers, text='Dropout rate:').grid(row=2, column=1, sticky=tk.N+tk.S+tk.E+tk.W)
        self.e_drop_rate = tk.Entry(self.tl_util_layers, textvariable=self.controller.layers_menu.s_drop_rate).grid(row=2, column=2, sticky=tk.N+tk.S+tk.E+tk.W)

        self.l_dense_size = tk.Label(self.tl_util_layers, text='Neurons:').grid(row=3, column=1, sticky=tk.N+tk.S+tk.E+tk.W)
        self.e_dense_size = tk.Entry(self.tl_util_layers, textvariable=self.controller.layers_menu.s_dense_size).grid(row=3, column=2, sticky=tk.N+tk.S+tk.E+tk.W)

        self.l_momentum = tk.Label(self.tl_util_layers, text='Momentum:').grid(row=4, column=1, sticky=tk.N+tk.S+tk.E+tk.W)
        self.e_momentum = tk.Entry(self.tl_util_layers, textvariable=self.controller.layers_menu.s_momentum).grid(row=4, column=2, sticky=tk.N+tk.S+tk.E+tk.W)

        self.l_epsilon = tk.Label(self.tl_util_layers, text='Epsilon:').grid(row=5, column=1, sticky=tk.N+tk.S+tk.E+tk.W)
        self.e_epsilon = tk.Entry(self.tl_util_layers, textvariable=self.controller.layers_menu.s_epsilon).grid(row=5, column=2, sticky=tk.N+tk.S+tk.E+tk.W)

        self.l_activation_type = tk.Label(self.tl_util_layers, text='Activation type:').grid(row=6, column=1, sticky=tk.N+tk.S+tk.E+tk.W)
        self.om_activation_type = tk.OptionMenu(self.tl_util_layers, self.controller.layers_menu.s_activation_type, *self.controller.layers_menu.o_activation_type)
        self.om_activation_type.config()
        self.om_activation_type.grid(row=6, column=2, sticky=tk.N+tk.S+tk.E+tk.W)

        self.l_permute_dims = tk.Label(self.tl_util_layers, text='Permute dimensions:').grid(row=7, column=1, sticky=tk.N+tk.S+tk.E+tk.W)
        self.e_permute_dims = tk.Entry(self.tl_util_layers, textvariable=self.controller.layers_menu.s_permute_dims).grid(row=7, column=2, sticky=tk.N+tk.S+tk.E+tk.W)

        self.l_spatial_drop_rate = tk.Label(self.tl_util_layers, text='Spatial dropout rate:').grid(row=8, column=1, sticky=tk.N+tk.S+tk.E+tk.W)
        self.e_spatial_drop_rate = tk.Entry(self.tl_util_layers, textvariable=self.controller.layers_menu.s_spatial_drop_rate).grid(row=8, column=2, sticky=tk.N+tk.S+tk.E+tk.W)

        self.l_gauss_drop_rate = tk.Label(self.tl_util_layers, text='Gaussian dropout rate:').grid(row=10, column=1, sticky=tk.N+tk.S+tk.E+tk.W)
        self.e_gauss_drop_rate = tk.Entry(self.tl_util_layers, textvariable=self.controller.layers_menu.s_gauss_drop_rate).grid(row=10, column=2, sticky=tk.N+tk.S+tk.E+tk.W)

        self.l_alpha_drop_rate = tk.Label(self.tl_util_layers, text='Alpha dropout rate:').grid(row=11, column=1, sticky=tk.N+tk.S+tk.E+tk.W)
        self.e_alpha_drop_rate = tk.Entry(self.tl_util_layers, textvariable=self.controller.layers_menu.s_alpha_drop_rate).grid(row=11, column=2, sticky=tk.N+tk.S+tk.E+tk.W)

        self.l_gauss_noise_std = tk.Label(self.tl_util_layers, text='Noise standard deviation:').grid(row=12, column=1, sticky=tk.N+tk.S+tk.E+tk.W)
        self.e_gauss_noise_std = tk.Entry(self.tl_util_layers, textvariable=self.controller.layers_menu.s_gauss_noise_std).grid(row=12, column=2, sticky=tk.N+tk.S+tk.E+tk.W)

        self.l_skip_type = tk.Label(self.tl_util_layers, text='Skip connection type:').grid(row=13, column=1, sticky=tk.N+tk.S+tk.E+tk.W)
        self.om_skip_type = tk.OptionMenu(self.tl_util_layers, self.controller.layers_menu.s_skip_type, *self.controller.layers_menu.o_skip_type)
        self.om_skip_type.config()
        self.om_skip_type.grid(row=13, column=2, sticky=tk.N+tk.S+tk.E+tk.W)
        self.tl_util_layers.withdraw()

    def show(self):
        self.tl_util_layers.deiconify()

    def b_input_click(self):
        layer = ':'.join(['Input', self.controller.layers_menu.s_input_size.get()])
        self.controller.layers_list_box.insert(tk.END, layer)

    def b_reshape_click(self):
        if any(self.controller.layers_list_box.get(0)) is False or self.controller.layers_list_box.get(0).split(':')[0] != 'Input':
            self.controller.errors_list_box.insert(tk.END, 'Level2Error:FirstLayerMustBeInput')
        else:
            layer = ':'.join(['Reshape', self.controller.layers_menu.s_reshape_dims.get()])
            self.controller.layers_list_box.insert(tk.END, layer)

    def b_dropout_click(self):
        if any(self.controller.layers_list_box.get(0)) is False or self.controller.layers_list_box.get(0).split(':')[0] != 'Input':
            self.controller.errors_list_box.insert(tk.END, 'Level2Error:FirstLayerMustBeInput')
        else:
            layer = ':'.join(['Dropout', self.controller.layers_menu.s_drop_rate.get()])
            self.controller.layers_list_box.insert(tk.END, layer)

    def b_dense_click(self):
        if any(self.controller.layers_list_box.get(0)) is False or self.controller.layers_list_box.get(0).split(':')[0] != 'Input':
            self.controller.errors_list_box.insert(tk.END, 'Level2Error:FirstLayerMustBeInput')
        else:
            layer = ':'.join(['Dense', self.controller.layers_menu.s_dense_size.get()])
            self.controller.layers_list_box.insert(tk.END, layer)

    def b_flatten_click(self):
        if any(self.controller.layers_list_box.get(0)) is False or self.controller.layers_list_box.get(0).split(':')[0] != 'Input':
            self.controller.errors_list_box.insert(tk.END, 'Level2Error:FirstLayerMustBeInput')
        else:
            layer = 'Flatten'
            self.controller.layers_list_box.insert(tk.END, layer)

    def b_batch_norm_click(self):
        if any(self.controller.layers_list_box.get(0)) is False or self.controller.layers_list_box.get(0).split(':')[0] != 'Input':
            self.controller.errors_list_box.insert(tk.END, 'Level2Error:FirstLayerMustBeInput')
        else:
            layer = ':'.join(['Batch normalization', self.controller.layers_menu.s_momentum.get(), self.controller.layers_menu.s_epsilon.get()])
            self.controller.layers_list_box.insert(tk.END, layer)

    def b_activation_click(self):
        if any(self.controller.layers_list_box.get(0)) is False or self.controller.layers_list_box.get(0).split(':')[0] != 'Input':
            self.controller.errors_list_box.insert(tk.END, 'Level2Error:FirstLayerMustBeInput')
        else:
            layer = ':'.join(['Activation', self.controller.layers_menu.s_activation_type.get()])
            self.controller.layers_list_box.insert(tk.END, layer)

    def b_permute_click(self):
        if any(self.controller.layers_list_box.get(0)) is False or self.controller.layers_list_box.get(0).split(':')[0] != 'Input':
            self.controller.errors_list_box.insert(tk.END, 'Level2Error:FirstLayerMustBeInput')
        else:
            layer = ':'.join(['Permute', self.controller.layers_menu.s_permute_dims.get()])
            self.controller.layers_list_box.insert(tk.END, layer)

    def b_2D_spatial_drop(self):
        if any(self.controller.layers_list_box.get(0)) is False or self.controller.layers_list_box.get(0).split(':')[0] != 'Input':
            self.controller.errors_list_box.insert(tk.END, 'Level2Error:FirstLayerMustBeInput')
        else:
            layer = ':'.join(['Spatial dropout 2D', self.controller.layers_menu.s_spatial_drop_rate.get()])
            self.controller.layers_list_box.insert(tk.END, layer)

    def b_3D_spatial_drop(self):
        if any(self.controller.layers_list_box.get(0)) is False or self.controller.layers_list_box.get(0).split(':')[0] != 'Input':
            self.controller.errors_list_box.insert(tk.END, 'Level2Error:FirstLayerMustBeInput')
        else:
            layer = ':'.join(['Spatial dropout 3D', self.controller.layers_menu.s_spatial_drop_rate.get()])
            self.controller.layers_list_box.insert(tk.END, layer)

    def b_gauss_drop_click(self):
        if any(self.controller.layers_list_box.get(0)) is False or self.controller.layers_list_box.get(0).split(':')[0] != 'Input':
            self.controller.errors_list_box.insert(tk.END, 'Level2Error:FirstLayerMustBeInput')
        else:
            layer = ':'.join(['Gaussian dropout', self.controller.layers_menu.s_gauss_drop_rate.get()])
            self.controller.layers_list_box.insert(tk.END, layer)

    def b_alpha_drop_click(self):
        if any(self.controller.layers_list_box.get(0)) is False or self.controller.layers_list_box.get(0).split(':')[0] != 'Input':
            self.controller.errors_list_box.insert(tk.END, 'Level2Error:FirstLayerMustBeInput')
        else:
            layer = ':'.join(['Alpha dropout', self.controller.layers_menu.s_alpha_drop_rate.get()])
            self.controller.layers_list_box.insert(tk.END, layer)

    def b_gauss_noise_click(self):
        if any(self.controller.layers_list_box.get(0)) is False or self.controller.layers_list_box.get(0).split(':')[0] != 'Input':
            self.controller.errors_list_box.insert(tk.END, 'Level2Error:FirstLayerMustBeInput')
        else:
            layer = ':'.join(['Gaussian noise', self.controller.layers_menu.s_gauss_noise_std.get()])
            self.controller.layers_list_box.insert(tk.END, layer)

    def b_out_skip_source_click(self):
        if any(self.controller.layers_list_box.get(0)) is False or self.controller.layers_list_box.get(0).split(':')[0] != 'Input':
            self.controller.errors_list_box.insert(tk.END, 'Level2Error:FirstLayerMustBeInput')
        else:
            layer = ':'.join(['Outer skip source', self.controller.layers_menu.s_skip_type.get()])
            self.controller.layers_list_box.insert(tk.END, layer)

    def b_out_skip_target_click(self):
        if any(self.controller.layers_list_box.get(0)) is False or self.controller.layers_list_box.get(0).split(':')[0] != 'Input':
            self.controller.errors_list_box.insert(tk.END, 'Level2Error:FirstLayerMustBeInput')
        else:
            layer = ':'.join(['Outer skip target', self.controller.layers_menu.s_skip_type.get()])
            self.controller.layers_list_box.insert(tk.END, layer)

    def b_in_skip_source_click(self):
        if any(self.controller.layers_list_box.get(0)) is False or self.controller.layers_list_box.get(0).split(':')[0] != 'Input':
            self.controller.errors_list_box.insert(tk.END, 'Level2Error:FirstLayerMustBeInput')
        else:
            layer = ':'.join(['Inner skip source', self.controller.layers_menu.s_skip_type.get()])
            self.controller.layers_list_box.insert(tk.END, layer)

    def b_in_skip_target_click(self):
        if any(self.controller.layers_list_box.get(0)) is False or self.controller.layers_list_box.get(0).split(':')[0] != 'Input':
            self.controller.errors_list_box.insert(tk.END, 'Level2Error:FirstLayerMustBeInput')
        else:
            layer = ':'.join(['Inner skip target', self.controller.layers_menu.s_skip_type.get()])
            self.controller.layers_list_box.insert(tk.END, layer)

    def b_hook_source_click(self):
        if any(self.controller.layers_list_box.get(0)) is False or self.controller.layers_list_box.get(0).split(':')[0] != 'Input':
            self.controller.errors_list_box.insert(tk.END, 'Level2Error:FirstLayerMustBeInput')
        else:
            layer = 'Hook connection source'
            self.controller.layers_list_box.insert(tk.END, layer)

    def b_remove_prev_click(self):
        self.controller.layers_list_box.delete(tk.END)
