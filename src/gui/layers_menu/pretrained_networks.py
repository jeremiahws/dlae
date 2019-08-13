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
"""src/gui/layers_menu/pretrained_networks.py"""


import tkinter as tk


class PretrainedNetworks:
    def __init__(self, controller):
        self.controller = controller
        self.button_heights = 1
        self.button_widths = 15
        self.label_heights = 1
        self.label_widths = 15
        self.entry_widths = 15

        self.tl_pretrained_nets = tk.Toplevel()
        self.tl_pretrained_nets.title('Pretrained networks')
        self.tl_pretrained_nets.resizable(width=False, height=False)
        self.tl_pretrained_nets.wm_protocol('WM_DELETE_WINDOW', self.tl_pretrained_nets.withdraw)
        self.b_vgg16 = tk.Button(self.tl_pretrained_nets, text='VGG16', command=self.b_vgg16_click).grid(row=0, column=0, sticky=tk.N+tk.S+tk.E+tk.W)
        self.b_vgg19 = tk.Button(self.tl_pretrained_nets, text='VGG19', command=self.b_vgg19_click).grid(row=0, column=1, sticky=tk.N+tk.S+tk.E+tk.W)
        self.b_resnet50 = tk.Button(self.tl_pretrained_nets, text='ResNet50', command=self.b_resnet50_click).grid(row=1, column=0, sticky=tk.N+tk.S+tk.E+tk.W)
        self.b_resnet101 = tk.Button(self.tl_pretrained_nets, text='ResNet101', command=self.b_resnet101_click).grid(row=1, column=1, sticky=tk.N+tk.S+tk.E+tk.W)
        self.b_resnet152 = tk.Button(self.tl_pretrained_nets, text='ResNet152', command=self.b_resnet152_click).grid(row=2, column=0, sticky=tk.N+tk.S+tk.E+tk.W)
        self.b_resnet50V2 = tk.Button(self.tl_pretrained_nets, text='ResNet50V2', command=self.b_resnet50v2_click).grid(row=2, column=1, sticky=tk.N+tk.S+tk.E+tk.W)
        self.b_resnet101V2 = tk.Button(self.tl_pretrained_nets, text='ResNet101V2', command=self.b_resnet101v2_click).grid(row=3, column=0, sticky=tk.N+tk.S+tk.E+tk.W)
        self.b_resnet152V2 = tk.Button(self.tl_pretrained_nets, text='ResNet152V2', command=self.b_resnet152v2_click).grid(row=3, column=1, sticky=tk.N+tk.S+tk.E+tk.W)
        self.b_resnext50 = tk.Button(self.tl_pretrained_nets, text='ResNeXt50', command=self.b_resnext50_click).grid(row=4, column=0, sticky=tk.N+tk.S+tk.E+tk.W)
        self.b_resnext101 = tk.Button(self.tl_pretrained_nets, text='ResNeXt101', command=self.b_resnext101_click).grid(row=4, column=1, sticky=tk.N+tk.S+tk.E+tk.W)
        self.b_inception_resnet_v2 = tk.Button(self.tl_pretrained_nets, text='InceptionResNetV2', command=self.b_inception_resnet_v2_click).grid(row=5, column=0, sticky=tk.N+tk.S+tk.E+tk.W)
        self.b_inceptiov_v3 = tk.Button(self.tl_pretrained_nets, text='InceptionV3', command=self.b_inception_v3_click).grid(row=5, column=1, sticky=tk.N+tk.S+tk.E+tk.W)
        self.b_densenet121 = tk.Button(self.tl_pretrained_nets, text='DenseNet121', command=self.b_densenet121_click).grid(row=6, column=0, sticky=tk.N+tk.S+tk.E+tk.W)
        self.b_densenet169 = tk.Button(self.tl_pretrained_nets, text='DenseNet169', command=self.b_densenet169_click).grid(row=6, column=1, sticky=tk.N+tk.S+tk.E+tk.W)
        self.b_densenet201 = tk.Button(self.tl_pretrained_nets, text='DenseNet201', command=self.b_densenet201_click).grid(row=7, column=0, sticky=tk.N+tk.S+tk.E+tk.W)
        self.b_xception = tk.Button(self.tl_pretrained_nets, text='Xception', command=self.b_xception_click).grid(row=7, column=1, sticky=tk.N+tk.S+tk.E+tk.W)
        self.b_mobilenet = tk.Button(self.tl_pretrained_nets, text='MobileNet', command=self.b_mobilenet_click).grid(row=8, column=0, sticky=tk.N+tk.S+tk.E+tk.W)
        self.b_mobilenet_v2 = tk.Button(self.tl_pretrained_nets, text='MobileNetV2', command=self.b_mobilenetv2_click).grid(row=8, column=1, sticky=tk.N+tk.S+tk.E+tk.W)

        self.l_include_top = tk.Label(self.tl_pretrained_nets, text='Include top:').grid(row=9, column=0, sticky=tk.N+tk.S+tk.E+tk.W)
        self.c_include_top = tk.Checkbutton(self.tl_pretrained_nets, variable=self.controller.layers_menu.bool_include_top).grid(row=9, column=1, sticky=tk.N+tk.S+tk.E+tk.W)

        self.l_weights_to_load = tk.Label(self.tl_pretrained_nets, text='Weights:').grid(row=10, column=0, sticky=tk.N+tk.S+tk.E+tk.W)
        self.om_weights_to_load = tk.OptionMenu(self.tl_pretrained_nets, self.controller.layers_menu.s_weights_to_load, *self.controller.layers_menu.o_weights_to_load)
        self.om_weights_to_load.config()
        self.om_weights_to_load.grid(row=10, column=1, sticky=tk.N+tk.S+tk.E+tk.W)

        self.l_include_skips = tk.Label(self.tl_pretrained_nets, text='Include skip connections:').grid(row=11, column=0, sticky=tk.N+tk.S+tk.E+tk.W)
        self.c_include_skips = tk.Checkbutton(self.tl_pretrained_nets, variable=self.controller.layers_menu.bool_include_skips).grid(row=11, column=1, sticky=tk.N+tk.S+tk.E+tk.W)

        self.l_include_hooks = tk.Label(self.tl_pretrained_nets, text='Include hook connections:').grid(row=12, column=0, sticky=tk.N+tk.S+tk.E+tk.W)
        self.c_include_hooks = tk.Checkbutton(self.tl_pretrained_nets, variable=self.controller.layers_menu.bool_include_hooks).grid(row=12, column=1, sticky=tk.N+tk.S+tk.E+tk.W)

        self.tl_pretrained_nets.withdraw()

    def show(self):
        self.tl_pretrained_nets.deiconify()

    def b_vgg16_click(self):
        if any(self.controller.layers_list_box.get(0)) is False or self.controller.layers_list_box.get(0).split(':')[0] != 'Input':
            self.controller.errors_list_box.insert(tk.END, 'Level2Error:FirstLayerMustBeInput')
        else:
            layer = ':'.join(['VGG16', str(self.controller.layers_menu.bool_include_top.get()), self.controller.layers_menu.s_weights_to_load.get(), self.controller.home_menu.s_input_shape.get(), str(self.controller.layers_menu.bool_include_skips.get()), str(self.controller.layers_menu.bool_include_hooks.get())])
            self.controller.layers_list_box.insert(tk.END, layer)

    def b_vgg19_click(self):
        if any(self.controller.layers_list_box.get(0)) is False or self.controller.layers_list_box.get(0).split(':')[0] != 'Input':
            self.controller.errors_list_box.insert(tk.END, 'Level2Error:FirstLayerMustBeInput')
        else:
            layer = ':'.join(['VGG19', str(self.controller.layers_menu.bool_include_top.get()), self.controller.layers_menu.s_weights_to_load.get(), self.controller.home_menu.s_input_shape.get(), str(self.controller.layers_menu.bool_include_skips.get()), str(self.controller.layers_menu.bool_include_hooks.get())])
            self.controller.layers_list_box.insert(tk.END, layer)

    def b_resnet50_click(self):
        if any(self.controller.layers_list_box.get(0)) is False or self.controller.layers_list_box.get(0).split(':')[0] != 'Input':
            self.controller.errors_list_box.insert(tk.END, 'Level2Error:FirstLayerMustBeInput')
        else:
            layer = ':'.join(['ResNet50', str(self.controller.layers_menu.bool_include_top.get()), self.controller.layers_menu.s_weights_to_load.get(), self.controller.home_menu.s_input_shape.get(), str(self.controller.layers_menu.bool_include_skips.get()), str(self.controller.layers_menu.bool_include_hooks.get())])
            self.controller.layers_list_box.insert(tk.END, layer)

    def b_resnet101_click(self):
        if any(self.controller.layers_list_box.get(0)) is False or self.controller.layers_list_box.get(0).split(':')[0] != 'Input':
            self.controller.errors_list_box.insert(tk.END, 'Level2Error:FirstLayerMustBeInput')
        else:
            layer = ':'.join(['ResNet101', str(self.controller.layers_menu.bool_include_top.get()), self.controller.layers_menu.s_weights_to_load.get(), self.controller.home_menu.s_input_shape.get(), str(self.controller.layers_menu.bool_include_skips.get()), str(self.controller.layers_menu.bool_include_hooks.get())])
            self.controller.layers_list_box.insert(tk.END, layer)

    def b_resnet152_click(self):
        if any(self.controller.layers_list_box.get(0)) is False or self.controller.layers_list_box.get(0).split(':')[0] != 'Input':
            self.controller.errors_list_box.insert(tk.END, 'Level2Error:FirstLayerMustBeInput')
        else:
            layer = ':'.join(['ResNet152', str(self.controller.layers_menu.bool_include_top.get()), self.controller.layers_menu.s_weights_to_load.get(), self.controller.home_menu.s_input_shape.get(), str(self.controller.layers_menu.bool_include_skips.get()), str(self.controller.layers_menu.bool_include_hooks.get())])
            self.controller.layers_list_box.insert(tk.END, layer)

    def b_resnet50v2_click(self):
        if any(self.controller.layers_list_box.get(0)) is False or self.controller.layers_list_box.get(0).split(':')[0] != 'Input':
            self.controller.errors_list_box.insert(tk.END, 'Level2Error:FirstLayerMustBeInput')
        else:
            layer = ':'.join(['ResNet50V2', str(self.controller.layers_menu.bool_include_top.get()), self.controller.layers_menu.s_weights_to_load.get(), self.controller.home_menu.s_input_shape.get(), str(self.controller.layers_menu.bool_include_skips.get()), str(self.controller.layers_menu.bool_include_hooks.get())])
            self.controller.layers_list_box.insert(tk.END, layer)

    def b_resnet101v2_click(self):
        if any(self.controller.layers_list_box.get(0)) is False or self.controller.layers_list_box.get(0).split(':')[0] != 'Input':
            self.controller.errors_list_box.insert(tk.END, 'Level2Error:FirstLayerMustBeInput')
        else:
            layer = ':'.join(['ResNet101V2', str(self.controller.layers_menu.bool_include_top.get()), self.controller.layers_menu.s_weights_to_load.get(), self.controller.home_menu.s_input_shape.get(), str(self.controller.layers_menu.bool_include_skips.get()), str(self.controller.layers_menu.bool_include_hooks.get())])
            self.controller.layers_list_box.insert(tk.END, layer)

    def b_resnet152v2_click(self):
        if any(self.controller.layers_list_box.get(0)) is False or self.controller.layers_list_box.get(0).split(':')[0] != 'Input':
            self.controller.errors_list_box.insert(tk.END, 'Level2Error:FirstLayerMustBeInput')
        else:
            layer = ':'.join(['ResNet152V2', str(self.controller.layers_menu.bool_include_top.get()), self.controller.layers_menu.s_weights_to_load.get(), self.controller.home_menu.s_input_shape.get(), str(self.controller.layers_menu.bool_include_skips.get()), str(self.controller.layers_menu.bool_include_hooks.get())])
            self.controller.layers_list_box.insert(tk.END, layer)

    def b_resnext50_click(self):
        if any(self.controller.layers_list_box.get(0)) is False or self.controller.layers_list_box.get(0).split(':')[0] != 'Input':
            self.controller.errors_list_box.insert(tk.END, 'Level2Error:FirstLayerMustBeInput')
        else:
            layer = ':'.join(['ResNeXt50', str(self.controller.layers_menu.bool_include_top.get()), self.controller.layers_menu.s_weights_to_load.get(), self.controller.home_menu.s_input_shape.get(), str(self.controller.layers_menu.bool_include_skips.get()), str(self.controller.layers_menu.bool_include_hooks.get())])
            self.controller.layers_list_box.insert(tk.END, layer)

    def b_resnext101_click(self):
        if any(self.controller.layers_list_box.get(0)) is False or self.controller.layers_list_box.get(0).split(':')[0] != 'Input':
            self.controller.errors_list_box.insert(tk.END, 'Level2Error:FirstLayerMustBeInput')
        else:
            layer = ':'.join(['ResNeXt101', str(self.controller.layers_menu.bool_include_top.get()), self.controller.layers_menu.s_weights_to_load.get(), self.controller.home_menu.s_input_shape.get(), str(self.controller.layers_menu.bool_include_skips.get()), str(self.controller.layers_menu.bool_include_hooks.get())])
            self.controller.layers_list_box.insert(tk.END, layer)

    def b_inception_resnet_v2_click(self):
        if any(self.controller.layers_list_box.get(0)) is False or self.controller.layers_list_box.get(0).split(':')[0] != 'Input':
            self.controller.errors_list_box.insert(tk.END, 'Level2Error:FirstLayerMustBeInput')
        else:
            layer = ':'.join(['InceptionResNetV2', str(self.controller.layers_menu.bool_include_top.get()), self.controller.layers_menu.s_weights_to_load.get(), self.controller.home_menu.s_input_shape.get(), str(self.controller.layers_menu.bool_include_skips.get()), str(self.controller.layers_menu.bool_include_hooks.get())])
            self.controller.layers_list_box.insert(tk.END, layer)

    def b_inception_v3_click(self):
        if any(self.controller.layers_list_box.get(0)) is False or self.controller.layers_list_box.get(0).split(':')[0] != 'Input':
            self.controller.errors_list_box.insert(tk.END, 'Level2Error:FirstLayerMustBeInput')
        else:
            layer = ':'.join(['InceptionV3', str(self.controller.layers_menu.bool_include_top.get()), self.controller.layers_menu.s_weights_to_load.get(), self.controller.home_menu.s_input_shape.get(), str(self.controller.layers_menu.bool_include_skips.get()), str(self.controller.layers_menu.bool_include_hooks.get())])
            self.controller.layers_list_box.insert(tk.END, layer)

    def b_densenet121_click(self):
        if any(self.controller.layers_list_box.get(0)) is False or self.controller.layers_list_box.get(0).split(':')[0] != 'Input':
            self.controller.errors_list_box.insert(tk.END, 'Level2Error:FirstLayerMustBeInput')
        else:
            layer = ':'.join(['DenseNet121', str(self.controller.layers_menu.bool_include_top.get()), self.controller.layers_menu.s_weights_to_load.get(), self.controller.home_menu.s_input_shape.get(), str(self.controller.layers_menu.bool_include_skips.get()), str(self.controller.layers_menu.bool_include_hooks.get())])
            self.controller.layers_list_box.insert(tk.END, layer)

    def b_densenet169_click(self):
        if any(self.controller.layers_list_box.get(0)) is False or self.controller.layers_list_box.get(0).split(':')[0] != 'Input':
            self.controller.errors_list_box.insert(tk.END, 'Level2Error:FirstLayerMustBeInput')
        else:
            layer = ':'.join(['DenseNet169', str(self.controller.layers_menu.bool_include_top.get()), self.controller.layers_menu.s_weights_to_load.get(), self.controller.home_menu.s_input_shape.get(), str(self.controller.layers_menu.bool_include_skips.get()), str(self.controller.layers_menu.bool_include_hooks.get())])
            self.controller.layers_list_box.insert(tk.END, layer)

    def b_densenet201_click(self):
        if any(self.controller.layers_list_box.get(0)) is False or self.controller.layers_list_box.get(0).split(':')[0] != 'Input':
            self.controller.errors_list_box.insert(tk.END, 'Level2Error:FirstLayerMustBeInput')
        else:
            layer = ':'.join(['DenseNet201', str(self.controller.layers_menu.bool_include_top.get()), self.controller.layers_menu.s_weights_to_load.get(), self.controller.home_menu.s_input_shape.get(), str(self.controller.layers_menu.bool_include_skips.get()), str(self.controller.layers_menu.bool_include_hooks.get())])
            self.controller.layers_list_box.insert(tk.END, layer)

    def b_xception_click(self):
        if any(self.controller.layers_list_box.get(0)) is False or self.controller.layers_list_box.get(0).split(':')[0] != 'Input':
            self.controller.errors_list_box.insert(tk.END, 'Level2Error:FirstLayerMustBeInput')
        else:
            layer = ':'.join(['Xception', str(self.controller.layers_menu.bool_include_top.get()), self.controller.layers_menu.s_weights_to_load.get(), self.controller.home_menu.s_input_shape.get(), str(self.controller.layers_menu.bool_include_skips.get()), str(self.controller.layers_menu.bool_include_hooks.get())])
            self.controller.layers_list_box.insert(tk.END, layer)

    def b_mobilenet_click(self):
        if any(self.controller.layers_list_box.get(0)) is False or self.controller.layers_list_box.get(0).split(':')[0] != 'Input':
            self.controller.errors_list_box.insert(tk.END, 'Level2Error:FirstLayerMustBeInput')
        else:
            layer = ':'.join(['MobileNet', str(self.controller.layers_menu.bool_include_top.get()), self.controller.layers_menu.s_weights_to_load.get(), self.controller.home_menu.s_input_shape.get(), str(self.controller.layers_menu.bool_include_skips.get()), str(self.controller.layers_menu.bool_include_hooks.get())])
            self.controller.layers_list_box.insert(tk.END, layer)

    def b_mobilenetv2_click(self):
        if any(self.controller.layers_list_box.get(0)) is False or self.controller.layers_list_box.get(0).split(':')[0] != 'Input':
            self.controller.errors_list_box.insert(tk.END, 'Level2Error:FirstLayerMustBeInput')
        else:
            layer = ':'.join(['MobileNetV2', str(self.controller.layers_menu.bool_include_top.get()), self.controller.layers_menu.s_weights_to_load.get(), self.controller.home_menu.s_input_shape.get(), str(self.controller.layers_menu.bool_include_skips.get()), str(self.controller.layers_menu.bool_include_hooks.get())])
            self.controller.layers_list_box.insert(tk.END, layer)
