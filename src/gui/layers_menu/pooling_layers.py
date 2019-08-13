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
"""src/gui/layers_menu/pooling_layers.py"""


import tkinter as tk


class PoolingLayers:
    def __init__(self, controller):
        self.controller = controller
        self.button_heights = 1
        self.button_widths = 15
        self.label_heights = 1
        self.label_widths = 15
        self.entry_widths = 15

        self.tl_pool_layers = tk.Toplevel()
        self.tl_pool_layers.title('Pooling layers')
        self.tl_pool_layers.resizable(width=False, height=False)
        self.tl_pool_layers.wm_protocol('WM_DELETE_WINDOW', self.tl_pool_layers.withdraw)
        self.b_max_pool_2D = tk.Button(self.tl_pool_layers, text='2D Maximum Pooling', command=self.b_2D_max_pool).grid(row=0, column=0, sticky=tk.N+tk.S+tk.E+tk.W)
        self.b_max_pool_3D = tk.Button(self.tl_pool_layers, text='3D Maximum Pooling', command=self.b_3D_max_pool).grid(row=0, column=1, sticky=tk.N+tk.S+tk.E+tk.W)
        self.b_avg_pool_2D = tk.Button(self.tl_pool_layers, text='2D Average Pooling', command=self.b_2D_avg_pool).grid(row=1, column=0, sticky=tk.N+tk.S+tk.E+tk.W)
        self.b_avg_pool_3D = tk.Button(self.tl_pool_layers, text='3D Average Pooling', command=self.b_3D_avg_pool).grid(row=1, column=1, sticky=tk.N+tk.S+tk.E+tk.W)
        self.b_global_max_pool_2D = tk.Button(self.tl_pool_layers, text='2D Global Maximum Pooling', command=self.b_2D_global_max_pool).grid(row=2, column=0, sticky=tk.N+tk.S+tk.E+tk.W)
        self.b_global_max_pool_3D = tk.Button(self.tl_pool_layers, text='3D Global Maximum Pooling', command=self.b_3D_global_max_pool).grid(row=2, column=1, sticky=tk.N+tk.S+tk.E+tk.W)
        self.b_global_avg_pool_2D = tk.Button(self.tl_pool_layers, text='2D Global Average Pooling', command=self.b_2D_global_avg_pool).grid(row=3, column=0, sticky=tk.N+tk.S+tk.E+tk.W)
        self.b_global_avg_pool_3D = tk.Button(self.tl_pool_layers, text='3D Global Average Pooling', command=self.b_3D_global_avg_pool).grid(row=3, column=1, sticky=tk.N+tk.S+tk.E+tk.W)

        self.l_pool_size = tk.Label(self.tl_pool_layers, text='Pooling size:').grid(row=0, column=2, sticky=tk.N+tk.S+tk.E+tk.W)
        self.e_pool_size = tk.Entry(self.tl_pool_layers, textvariable=self.controller.layers_menu.s_pool_size).grid(row=0, column=3, sticky=tk.N+tk.S+tk.E+tk.W)

        self.l_pool_stride = tk.Label(self.tl_pool_layers, text='Stride:').grid(row=1, column=2)
        self.e_pool_stride = tk.Entry(self.tl_pool_layers, textvariable=self.controller.layers_menu.s_pool_stride).grid(row=1, column=3, sticky=tk.N+tk.S+tk.E+tk.W)
        self.tl_pool_layers.withdraw()

    def show(self):
        self.tl_pool_layers.deiconify()

    def b_2D_max_pool(self):
        if any(self.controller.layers_list_box.get(0)) is False or self.controller.layers_list_box.get(0).split(':')[0] != 'Input':
            self.controller.errors_list_box.insert(tk.END, 'Level2Error:FirstLayerMustBeInput')
        else:
            layer = ':'.join(['Max pooling 2D', self.controller.layers_menu.s_pool_size.get(), self.controller.layers_menu.s_pool_stride.get()])
            self.controller.layers_list_box.insert(tk.END, layer)

    def b_3D_max_pool(self):
        if any(self.controller.layers_list_box.get(0)) is False or self.controller.layers_list_box.get(0).split(':')[0] != 'Input':
            self.controller.errors_list_box.insert(tk.END, 'Level2Error:FirstLayerMustBeInput')
        else:
            layer = ':'.join(['Max pooling 3D', self.controller.layers_menu.s_pool_size.get(), self.controller.layers_menu.s_pool_stride.get()])
            self.controller.layers_list_box.insert(tk.END, layer)

    def b_2D_avg_pool(self):
        if any(self.controller.layers_list_box.get(0)) is False or self.controller.layers_list_box.get(0).split(':')[0] != 'Input':
            self.controller.errors_list_box.insert(tk.END, 'Level2Error:FirstLayerMustBeInput')
        else:
            layer = ':'.join(['Average pooling 2D', self.controller.layers_menu.s_pool_size.get(), self.controller.layers_menu.s_pool_stride.get()])
            self.controller.layers_list_box.insert(tk.END, layer)

    def b_3D_avg_pool(self):
        if any(self.controller.layers_list_box.get(0)) is False or self.controller.layers_list_box.get(0).split(':')[0] != 'Input':
            self.controller.errors_list_box.insert(tk.END, 'Level2Error:FirstLayerMustBeInput')
        else:
            layer = ':'.join(['Average pooling 3D', self.controller.layers_menu.s_pool_size.get(), self.controller.layers_menu.s_pool_stride.get()])
            self.controller.layers_list_box.insert(tk.END, layer)

    def b_2D_global_max_pool(self):
        if any(self.controller.layers_list_box.get(0)) is False or self.controller.layers_list_box.get(0).split(':')[0] != 'Input':
            self.controller.errors_list_box.insert(tk.END, 'Level2Error:FirstLayerMustBeInput')
        else:
            layer = 'Global max pooling 2D'
            self.controller.layers_list_box.insert(tk.END, layer)

    def b_3D_global_max_pool(self):
        if any(self.controller.layers_list_box.get(0)) is False or self.controller.layers_list_box.get(0).split(':')[0] != 'Input':
            self.controller.errors_list_box.insert(tk.END, 'Level2Error:FirstLayerMustBeInput')
        else:
            layer = 'Global max pooling 3D'
            self.controller.layers_list_box.insert(tk.END, layer)

    def b_2D_global_avg_pool(self):
        if any(self.controller.layers_list_box.get(0)) is False or self.controller.layers_list_box.get(0).split(':')[0] != 'Input':
            self.controller.errors_list_box.insert(tk.END, 'Level2Error:FirstLayerMustBeInput')
        else:
            layer = 'Global average pooling 2D'
            self.controller.layers_list_box.insert(tk.END, layer)

    def b_3D_global_avg_pool(self):
        if any(self.controller.layers_list_box.get(0)) is False or self.controller.layers_list_box.get(0).split(':')[0] != 'Input':
            self.controller.errors_list_box.insert(tk.END, 'Level2Error:FirstLayerMustBeInput')
        else:
            layer = 'Global average pooling 3D'
            self.controller.layers_list_box.insert(tk.END, layer)
