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
"""src/gui/layers_menu/convolutional_layers.py"""


import tkinter as tk


class ConvolutionalLayers:
    def __init__(self, controller):
        self.controller = controller
        self.button_heights = 1
        self.button_widths = 15
        self.label_heights = 1
        self.label_widths = 15
        self.entry_widths = 15
        
        self.tl_conv_layers = tk.Toplevel()
        self.tl_conv_layers.title('Convolution layers')
        self.tl_conv_layers.resizable(width=False, height=False)
        self.tl_conv_layers.wm_protocol('WM_DELETE_WINDOW', self.tl_conv_layers.withdraw)
        self.b_conv_2D = tk.Button(self.tl_conv_layers, text='2D Conv', command=self.b_2D_conv, height=self.button_heights, width=self.button_widths).grid(row=0, column=0, sticky=tk.N+tk.S+tk.E+tk.W)
        self.b_conv_3D = tk.Button(self.tl_conv_layers, text='3D Conv', command=self.b_3D_conv, height=self.button_heights, width=self.button_widths).grid(row=0, column=1, sticky=tk.N+tk.S+tk.E+tk.W)
        self.b_transpose_conv_2D = tk.Button(self.tl_conv_layers, text='2D Transpose Conv', command=self.b_2D_transpose_conv, height=self.button_heights, width=self.button_widths).grid(row=1, column=0, sticky=tk.N+tk.S+tk.E+tk.W)
        self.b_transpose_conv_3D = tk.Button(self.tl_conv_layers, text='3D Transpose Conv', command=self.b_3D_transpose_conv, height=self.button_heights, width=self.button_widths).grid(row=1, column=1, sticky=tk.N+tk.S+tk.E+tk.W)
        self.b_resize_conv_2D = tk.Button(self.tl_conv_layers, text='2D Resize Conv', command=self.b_2D_resize_conv, height=self.button_heights, width=self.button_widths).grid(row=2, column=0, sticky=tk.N+tk.S+tk.E+tk.W)
        self.b_resize_conv_3D = tk.Button(self.tl_conv_layers, text='3D Resize Conv', command=self.b_3D_resize_conv, height=self.button_heights, width=self.button_widths).grid(row=2, column=1, sticky=tk.N+tk.S+tk.E+tk.W)
        self.b_sep_conv_2D = tk.Button(self.tl_conv_layers, text='2D Separable Conv', command=self.b_2D_sep_conv, height=self.button_heights, width=self.button_widths).grid(row=3, column=0, sticky=tk.N+tk.S+tk.E+tk.W)
        self.b_sep_conv_3D = tk.Button(self.tl_conv_layers, text='3D Separable Conv', command=self.b_3D_sep_conv, height=self.button_heights, width=self.button_widths).grid(row=3, column=1, sticky=tk.N+tk.S+tk.E+tk.W)
        self.b_depth_conv_2D = tk.Button(self.tl_conv_layers, text='2D Depthwise Conv', command=self.b_2D_depth_conv, height=self.button_heights, width=self.button_widths).grid(row=4, column=0, sticky=tk.N+tk.S+tk.E+tk.W)
        self.b_depth_conv_3D = tk.Button(self.tl_conv_layers, text='3D Depthwise Conv', command=self.b_3D_depth_conv, height=self.button_heights, width=self.button_widths).grid(row=4, column=1, sticky=tk.N+tk.S+tk.E+tk.W)
        self.b_upsample_2D = tk.Button(self.tl_conv_layers, text='2D Upsample', command=self.b_2D_upsample, height=self.button_heights, width=self.button_widths).grid(row=5, column=0, sticky=tk.N+tk.S+tk.E+tk.W)
        self.b_upsample_3D = tk.Button(self.tl_conv_layers, text='3D Upsample', command=self.b_3D_upsample, height=self.button_heights, width=self.button_widths).grid(row=5, column=1, sticky=tk.N+tk.S+tk.E+tk.W)
        self.b_zero_pad_2D = tk.Button(self.tl_conv_layers, text='2D Zero Padding', command=self.b_2D_zero_pad, height=self.button_heights, width=self.button_widths).grid(row=6, column=0, sticky=tk.N+tk.S+tk.E+tk.W)
        self.b_zero_pad_3D = tk.Button(self.tl_conv_layers, text='3D Zero Padding', command=self.b_3D_zero_pad, height=self.button_heights, width=self.button_widths).grid(row=6, column=1, sticky=tk.N+tk.S+tk.E+tk.W)
        self.b_cropping_2D = tk.Button(self.tl_conv_layers, text='2D Cropping', command=self.b_2D_cropping, height=self.button_heights, width=self.button_widths).grid(row=7, column=0, sticky=tk.N+tk.S+tk.E+tk.W)
        self.b_cropping_3D = tk.Button(self.tl_conv_layers, text='3D Cropping', command=self.b_3D_cropping, height=self.button_heights, width=self.button_widths).grid(row=7, column=1, sticky=tk.N+tk.S+tk.E+tk.W)

        self.l_padding = tk.Label(self.tl_conv_layers, text='Padding:', height=self.label_heights, width=self.label_widths).grid(row=8, column=0, sticky=tk.N+tk.S+tk.E+tk.W)
        self.om_padding = tk.OptionMenu(self.tl_conv_layers, self.controller.layers_menu.s_padding, *self.controller.layers_menu.o_padding)
        self.om_padding.config()
        self.om_padding.grid(row=8, column=1, sticky=tk.N+tk.S+tk.E+tk.W)

        self.l_initializer = tk.Label(self.tl_conv_layers, text='Kernel initializer:', height=self.label_heights, width=self.label_widths).grid(row=9, column=0, sticky=tk.N+tk.S+tk.E+tk.W)
        self.om_initializer = tk.OptionMenu(self.tl_conv_layers, self.controller.layers_menu.s_initializer, *self.controller.layers_menu.o_initializer)
        self.om_initializer.config()
        self.om_initializer.grid(row=9, column=1, sticky=tk.N+tk.S+tk.E+tk.W)

        self.l_kernel_regularizer = tk.Label(self.tl_conv_layers, text='Kernel regularizer:', height=self.label_heights, width=self.label_widths).grid(row=10, column=0, sticky=tk.N+tk.S+tk.E+tk.W)
        self.om_kernel_regularizer = tk.OptionMenu(self.tl_conv_layers, self.controller.layers_menu.s_kernel_regularizer, *self.controller.layers_menu.o_kernel_regularizer)
        self.om_kernel_regularizer.config()
        self.om_kernel_regularizer.grid(row=10, column=1, sticky=tk.N+tk.S+tk.E+tk.W)

        self.l_activity_regularizer = tk.Label(self.tl_conv_layers, text='Activity regularizer:', height=self.label_heights, width=self.label_widths).grid(row=11, column=0, sticky=tk.N+tk.S+tk.E+tk.W)
        self.om_activity_regularizer = tk.OptionMenu(self.tl_conv_layers, self.controller.layers_menu.s_activity_regularizer, *self.controller.layers_menu.o_activity_regularizer)
        self.om_activity_regularizer.config()
        self.om_activity_regularizer.grid(row=11, column=1, sticky=tk.N+tk.S+tk.E+tk.W)

        self.l_conv_layer_maps = tk.Label(self.tl_conv_layers, text='Maps:', height=self.label_heights, width=self.label_widths).grid(row=0, column=2, sticky=tk.N+tk.S+tk.E+tk.W)
        self.e_conv_layer_maps = tk.Entry(self.tl_conv_layers, textvariable=self.controller.layers_menu.s_conv_layer_maps,  width=self.entry_widths).grid(row=0, column=3)

        self.l_conv_layer_kernel = tk.Label(self.tl_conv_layers, text='Kernel:', height=self.label_heights, width=self.label_widths).grid(row=1, column=2, sticky=tk.N+tk.S+tk.E+tk.W)
        self.e_conv_layer_kernel = tk.Entry(self.tl_conv_layers, textvariable=self.controller.layers_menu.s_conv_layer_kernel, width=self.entry_widths).grid(row=1, column=3)

        self.l_conv_layer_stride = tk.Label(self.tl_conv_layers, text='Stride:', height=self.label_heights, width=self.label_widths).grid(row=2, column=2, sticky=tk.N+tk.S+tk.E+tk.W)
        self.e_conv_layer_stride = tk.Entry(self.tl_conv_layers, textvariable=self.controller.layers_menu.s_conv_layer_stride, width=self.entry_widths).grid(row=2, column=3)

        self.l_r_conv_layer_upsample = tk.Label(self.tl_conv_layers, text='Resize:', height=self.label_heights, width=self.label_widths).grid(row=3, column=2, sticky=tk.N+tk.S+tk.E+tk.W)
        self.e_r_conv_layer_upsample = tk.Entry(self.tl_conv_layers, textvariable=self.controller.layers_menu.s_r_conv_layer_upsample, width=self.entry_widths).grid(row=3, column=3)

        self.l_dilation_rate = tk.Label(self.tl_conv_layers, text='Dilation rate:', height=self.label_heights, width=self.label_widths).grid(row=4, column=2, sticky=tk.N+tk.S+tk.E+tk.W)
        self.e_dilation_rate = tk.Entry(self.tl_conv_layers, textvariable=self.controller.layers_menu.s_dilation_rate, width=self.entry_widths).grid(row=4, column=3)

        self.l_upsample_size = tk.Label(self.tl_conv_layers, text='Upsample size:', height=self.label_heights, width=self.label_widths).grid(row=5, column=2, sticky=tk.N+tk.S+tk.E+tk.W)
        self.e_upsample_size = tk.Entry(self.tl_conv_layers, textvariable=self.controller.layers_menu.s_upsample_size, width=self.entry_widths).grid(row=5, column=3)

        self.l_zeropad = tk.Label(self.tl_conv_layers, text='Zero padding:', height=self.label_heights, width=self.label_widths).grid(row=6, column=2, sticky=tk.N+tk.S+tk.E+tk.W)
        self.e_zeropad = tk.Entry(self.tl_conv_layers, textvariable=self.controller.layers_menu.s_zeropad, width=self.entry_widths).grid(row=6, column=3)

        self.l_cropping = tk.Label(self.tl_conv_layers, text='Cropping:', height=self.label_heights, width=self.label_widths).grid(row=7, column=2, sticky=tk.N+tk.S+tk.E+tk.W)
        self.e_cropping = tk.Entry(self.tl_conv_layers, textvariable=self.controller.layers_menu.s_cropping, width=self.entry_widths).grid(row=7, column=3)

        self.l_l1 = tk.Label(self.tl_conv_layers, text='l1:', height=self.label_heights, width=self.label_widths).grid(row=10, column=2, sticky=tk.N+tk.S+tk.E+tk.W)
        self.e_l1 = tk.Entry(self.tl_conv_layers, textvariable=self.controller.layers_menu.s_l1, width=self.entry_widths).grid(row=10, column=3)

        self.l_l2 = tk.Label(self.tl_conv_layers, text='l2:', height=self.label_heights, width=self.label_widths).grid(row=11, column=2, sticky=tk.N+tk.S+tk.E+tk.W)
        self.e_l2 = tk.Entry(self.tl_conv_layers, textvariable=self.controller.layers_menu.s_l2, width=self.entry_widths).grid(row=11, column=3)
        self.tl_conv_layers.withdraw()

    def show(self):
        self.tl_conv_layers.deiconify()

    def b_2D_conv(self):
        if any(self.controller.layers_list_box.get(0)) is False or self.controller.layers_list_box.get(0).split(':')[0] != 'Input':
            self.controller.errors_list_box.insert(tk.END, 'Level2Error:FirstLayerMustBeInput')
        else:
            layer = ':'.join(['Convolution 2D', self.controller.layers_menu.s_conv_layer_maps.get(), self.controller.layers_menu.s_conv_layer_kernel.get(),
                              self.controller.layers_menu.s_conv_layer_stride.get(), self.controller.layers_menu.s_padding.get(), self.controller.layers_menu.s_dilation_rate.get(),
                              self.controller.layers_menu.s_initializer.get(), self.controller.layers_menu.s_kernel_regularizer.get(),
                              self.controller.layers_menu.s_activity_regularizer.get(), self.controller.layers_menu.s_l1.get(), self.controller.layers_menu.s_l2.get()])
            self.controller.layers_list_box.insert(tk.END, layer)

    def b_3D_conv(self):
        if any(self.controller.layers_list_box.get(0)) is False or self.controller.layers_list_box.get(0).split(':')[0] != 'Input':
            self.controller.errors_list_box.insert(tk.END, 'Level2Error:FirstLayerMustBeInput')
        else:
            layer = ':'.join(['Convolution 3D', self.controller.layers_menu.s_conv_layer_maps.get(), self.controller.layers_menu.s_conv_layer_kernel.get(),
                              self.controller.layers_menu.s_conv_layer_stride.get(), self.controller.layers_menu.s_padding.get(), self.controller.layers_menu.s_dilation_rate.get(),
                              self.controller.layers_menu.s_initializer.get(), self.controller.layers_menu.s_kernel_regularizer.get(),
                              self.controller.layers_menu.s_activity_regularizer.get(), self.controller.layers_menu.s_l1.get(), self.controller.layers_menu.s_l2.get()])
            self.controller.layers_list_box.insert(tk.END, layer)

    def b_2D_transpose_conv(self):
        if any(self.controller.layers_list_box.get(0)) is False or self.controller.layers_list_box.get(0).split(':')[0] != 'Input':
            self.controller.errors_list_box.insert(tk.END, 'Level2Error:FirstLayerMustBeInput')
        else:
            layer = ':'.join(['Transpose convolution 2D', self.controller.layers_menu.s_conv_layer_maps.get(), self.controller.layers_menu.s_conv_layer_kernel.get(),
                              self.controller.layers_menu.s_conv_layer_stride.get(), self.controller.layers_menu.s_padding.get(), self.controller.layers_menu.s_dilation_rate.get(),
                              self.controller.layers_menu.s_initializer.get(), self.controller.layers_menu.s_kernel_regularizer.get(),
                              self.controller.layers_menu.s_activity_regularizer.get(), self.controller.layers_menu.s_l1.get(), self.controller.layers_menu.s_l2.get()])
            self.controller.layers_list_box.insert(tk.END, layer)

    def b_3D_transpose_conv(self):
        if any(self.controller.layers_list_box.get(0)) is False or self.controller.layers_list_box.get(0).split(':')[0] != 'Input':
            self.controller.errors_list_box.insert(tk.END, 'Level2Error:FirstLayerMustBeInput')
        else:
            layer = ':'.join(['Transpose convolution 3D', self.controller.layers_menu.s_conv_layer_maps.get(), self.controller.layers_menu.s_conv_layer_kernel.get(),
                              self.controller.layers_menu.s_conv_layer_stride.get(), self.controller.layers_menu.s_padding.get(), self.controller.layers_menu.s_dilation_rate.get(),
                              self.controller.layers_menu.s_initializer.get(), self.controller.layers_menu.s_kernel_regularizer.get(),
                              self.controller.layers_menu.s_activity_regularizer.get(), self.controller.layers_menu.s_l1.get(), self.controller.layers_menu.s_l2.get()])
            self.controller.layers_list_box.insert(tk.END, layer)

    def b_2D_resize_conv(self):
        if any(self.controller.layers_list_box.get(0)) is False or self.controller.layers_list_box.get(0).split(':')[0] != 'Input':
            self.controller.errors_list_box.insert(tk.END, 'Level2Error:FirstLayerMustBeInput')
        else:
            layer = ':'.join(['Resize convolution 2D', self.controller.layers_menu.s_conv_layer_maps.get(), self.controller.layers_menu.s_conv_layer_kernel.get(),
                              self.controller.layers_menu.s_conv_layer_stride.get(), self.controller.layers_menu.s_r_conv_layer_upsample.get(), self.controller.layers_menu.s_padding.get(),
                              self.controller.layers_menu.s_dilation_rate.get(), self.controller.layers_menu.s_initializer.get(), self.controller.layers_menu.s_kernel_regularizer.get(),
                              self.controller.layers_menu.s_activity_regularizer.get(), self.controller.layers_menu.s_l1.get(), self.controller.layers_menu.s_l2.get()])
            self.controller.layers_list_box.insert(tk.END, layer)

    def b_3D_resize_conv(self):
        if any(self.controller.layers_list_box.get(0)) is False or self.controller.layers_list_box.get(0).split(':')[0] != 'Input':
            self.controller.errors_list_box.insert(tk.END, 'Level2Error:FirstLayerMustBeInput')
        else:
            layer = ':'.join(['Resize convolution 3D', self.controller.layers_menu.s_conv_layer_maps.get(), self.controller.layers_menu.s_conv_layer_kernel.get(),
                              self.controller.layers_menu.s_conv_layer_stride.get(), self.controller.layers_menu.s_r_conv_layer_upsample.get(), self.controller.layers_menu.s_padding.get(),
                              self.controller.layers_menu.s_dilation_rate.get(), self.controller.layers_menu.s_initializer.get(), self.controller.layers_menu.s_kernel_regularizer.get(),
                              self.controller.layers_menu.s_activity_regularizer.get(), self.controller.layers_menu.s_l1.get(), self.controller.layers_menu.s_l2.get()])
            self.controller.layers_list_box.insert(tk.END, layer)

    def b_2D_sep_conv(self):
        if any(self.controller.layers_list_box.get(0)) is False or self.controller.layers_list_box.get(0).split(':')[0] != 'Input':
            self.controller.errors_list_box.insert(tk.END, 'Level2Error:FirstLayerMustBeInput')
        else:
            layer = ':'.join(['Separable convolution 2D', self.controller.layers_menu.s_conv_layer_maps.get(), self.controller.layers_menu.s_conv_layer_kernel.get(),
                              self.controller.layers_menu.s_conv_layer_stride.get(), self.controller.layers_menu.s_padding.get(), self.controller.layers_menu.s_dilation_rate.get(),
                              self.controller.layers_menu.s_initializer.get(), self.controller.layers_menu.s_kernel_regularizer.get(),
                              self.controller.layers_menu.s_activity_regularizer.get(), self.controller.layers_menu.s_l1.get(), self.controller.layers_menu.s_l2.get()])
            self.controller.layers_list_box.insert(tk.END, layer)

    def b_3D_sep_conv(self):
        #TODO uncomment once 3D separable convs are implemented
        # if any(self.controller.layers_list_box.get(0)) is False or self.controller.layers_list_box.get(0).split(':')[0] != 'Input':
        #     self.controller.errors_list_box.insert(tk.END, 'Level2Error:FirstLayerMustBeInput')
        # else:
        #     layer = ':'.join(['Separable convolution 2D', self.controller.layers_menu.s_conv_layer_maps.get(), self.controller.layers_menu.s_conv_layer_kernel.get(),
        #                       self.controller.layers_menu.s_conv_layer_stride.get(), self.controller.layers_menu.s_padding.get(), self.controller.layers_menu.s_dilation_rate.get(),
        #                       self.controller.layers_menu.s_initializer.get(), self.controller.layers_menu.s_kernel_regularizer.get(),
        #                       self.controller.layers_menu.s_activity_regularizer.get(), self.controller.layers_menu.s_l1.get(), self.controller.layers_menu.s_l2.get()])
        #     self.controller.layers_list_box.insert(tk.END, layer)
        self.controller.errors_list_box.insert(tk.END, 'Level2Warning:3DSeparableConvolutionsNotCurrentlySupported')

    def b_2D_depth_conv(self):
        if any(self.controller.layers_list_box.get(0)) is False or self.controller.layers_list_box.get(0).split(':')[0] != 'Input':
            self.controller.errors_list_box.insert(tk.END, 'Level2Error:FirstLayerMustBeInput')
        else:
            layer = ':'.join(
                ['Depthwise separable convolution 2D', self.controller.layers_menu.s_conv_layer_maps.get(), self.controller.layers_menu.s_conv_layer_kernel.get(),
                 self.controller.layers_menu.s_conv_layer_stride.get(), self.controller.layers_menu.s_padding.get(),
                 self.controller.layers_menu.s_initializer.get(), self.controller.layers_menu.s_kernel_regularizer.get(),
                 self.controller.layers_menu.s_activity_regularizer.get(), self.controller.layers_menu.s_l1.get(), self.controller.layers_menu.s_l2.get()])
            self.controller.layers_list_box.insert(tk.END, layer)

    def b_3D_depth_conv(self):
        # TODO uncomment once 3D depthwise separable convs are implemented
        # if any(self.controller.layers_list_box.get(0)) is False or self.controller.layers_list_box.get(0).split(':')[0] != 'Input':
        #     self.controller.errors_list_box.insert(tk.END, 'Level2Error:FirstLayerMustBeInput')
        # else:
        #     layer = ':'.join(
        #         ['Depthwise separable convolution 3D', self.controller.layers_menu.s_conv_layer_maps.get(), self.controller.layers_menu.s_conv_layer_kernel.get(),
        #          self.controller.layers_menu.s_conv_layer_stride.get(), self.controller.layers_menu.s_padding.get(),
        #          self.controller.layers_menu.s_initializer.get(), self.controller.layers_menu.s_kernel_regularizer.get(), self.controller.layers_menu.s_activity_regularizer.get(),
        #          self.controller.layers_menu.s_l1.get(), self.controller.layers_menu.s_l2.get()])
        #     self.controller.layers_list_box.insert(tk.END, layer)
        self.controller.errors_list_box.insert(tk.END, 'Level2Warning:3DDepthwiseSeparableConvolutionsNotCurrentlySupported')

    def b_2D_upsample(self):
        if any(self.controller.layers_list_box.get(0)) is False or self.controller.layers_list_box.get(0).split(':')[0] != 'Input':
            self.controller.errors_list_box.insert(tk.END, 'Level2Error:FirstLayerMustBeInput')
        else:
            layer = ':'.join(['Upsample 2D', self.controller.layers_menu.s_upsample_size.get()])
            self.controller.layers_list_box.insert(tk.END, layer)

    def b_3D_upsample(self):
        if any(self.controller.layers_list_box.get(0)) is False or self.controller.layers_list_box.get(0).split(':')[0] != 'Input':
            self.controller.errors_list_box.insert(tk.END, 'Level2Error:FirstLayerMustBeInput')
        else:
            layer = ':'.join(['Upsample 3D', self.controller.layers_menu.s_upsample_size.get()])
            self.controller.layers_list_box.insert(tk.END, layer)

    def b_2D_zero_pad(self):
        if any(self.controller.layers_list_box.get(0)) is False or self.controller.layers_list_box.get(0).split(':')[0] != 'Input':
            self.controller.errors_list_box.insert(tk.END, 'Level2Error:FirstLayerMustBeInput')
        else:
            layer = ':'.join(['Zero padding 2D', self.controller.layers_menu.s_zeropad.get()])
            self.controller.layers_list_box.insert(tk.END, layer)

    def b_3D_zero_pad(self):
        if any(self.controller.layers_list_box.get(0)) is False or self.controller.layers_list_box.get(0).split(':')[0] != 'Input':
            self.controller.errors_list_box.insert(tk.END, 'Level2Error:FirstLayerMustBeInput')
        else:
            layer = ':'.join(['Zero padding 3D', self.controller.layers_menu.s_zeropad.get()])
            self.controller.layers_list_box.insert(tk.END, layer)

    def b_2D_cropping(self):
        if any(self.controller.layers_list_box.get(0)) is False or self.controller.layers_list_box.get(0).split(':')[0] != 'Input':
            self.controller.errors_list_box.insert(tk.END, 'Level2Error:FirstLayerMustBeInput')
        else:
            layer = ':'.join(['Cropping 2D', self.controller.layers_menu.s_cropping.get()])
            self.controller.layers_list_box.insert(tk.END, layer)

    def b_3D_cropping(self):
        if any(self.controller.layers_list_box.get(0)) is False or self.controller.layers_list_box.get(0).split(':')[0] != 'Input':
            self.controller.errors_list_box.insert(tk.END, 'Level2Error:FirstLayerMustBeInput')
        else:
            layer = ':'.join(['Cropping 3D', self.controller.layers_menu.s_cropping.get()])
            self.controller.layers_list_box.insert(tk.END, layer)
