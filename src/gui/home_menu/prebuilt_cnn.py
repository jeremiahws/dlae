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
"""src/gui/home_menu/prebuilt_cnn.py"""


import tkinter as tk
import os
from src.utils.general_utils import load_config
import tensorflow as tf


class PrebuiltConvolutionalNeuralNetwork:
    def __init__(self, controller):
        self.tl_prebuilt_cnn = tk.Toplevel()
        self.tl_prebuilt_cnn.title('Prebuilt CNNs')
        self.tl_prebuilt_cnn.wm_protocol('WM_DELETE_WINDOW', self.tl_prebuilt_cnn.withdraw)
        self.controller = controller

        self.p_lenet5 = tk.PhotoImage(file='src/gui/button_figs/prebuilt_cnn/lenet5.png')
        self.b_lenet5 = tk.Button(self.tl_prebuilt_cnn, image=self.p_lenet5, command=self.lenet5).grid(row=0, column=0)

        self.p_alexnet = tk.PhotoImage(file='src/gui/button_figs/prebuilt_cnn/alexnet.png')
        self.b_alexnet = tk.Button(self.tl_prebuilt_cnn, image=self.p_alexnet, command=self.alexnet).grid(row=0,
                                                                                                          column=1)
        self.p_vgg16 = tk.PhotoImage(file='src/gui/button_figs/prebuilt_cnn/vgg16.png')
        self.b_vgg16 = tk.Button(self.tl_prebuilt_cnn, image=self.p_vgg16, command=self.vgg16).grid(row=0, column=2)

        self.p_vgg19 = tk.PhotoImage(file='src/gui/button_figs/prebuilt_cnn/vgg19.png')
        self.b_vgg19 = tk.Button(self.tl_prebuilt_cnn, image=self.p_vgg19, command=self.vgg19).grid(row=0, column=3)

        self.p_resnet50 = tk.PhotoImage(file='src/gui/button_figs/prebuilt_cnn/resnet50.png')
        self.b_resnet50 = tk.Button(self.tl_prebuilt_cnn, image=self.p_resnet50, command=self.resnet50).grid(row=1, column=0)

        self.p_resnet101 = tk.PhotoImage(file='src/gui/button_figs/prebuilt_cnn/resnet101.png')
        self.b_resnet101 = tk.Button(self.tl_prebuilt_cnn, image=self.p_resnet101, command=self.resnet101).grid(row=1, column=1)

        self.p_resnet152 = tk.PhotoImage(file='src/gui/button_figs/prebuilt_cnn/resnet152.png')
        self.b_resnet152 = tk.Button(self.tl_prebuilt_cnn, image=self.p_resnet152, command=self.resnet152).grid(row=1, column=2)

        self.p_resnet50v2 = tk.PhotoImage(file='src/gui/button_figs/prebuilt_cnn/resnet50v2.png')
        self.b_resnet50v2 = tk.Button(self.tl_prebuilt_cnn, image=self.p_resnet50v2, command=self.resnet50v2).grid(row=1, column=3)

        self.p_resnet101v2 = tk.PhotoImage(file='src/gui/button_figs/prebuilt_cnn/resnet101v2.png')
        self.b_resnet101v2 = tk.Button(self.tl_prebuilt_cnn, image=self.p_resnet101v2, command=self.resnet101v2).grid(row=2, column=0)

        self.p_resnet152v2 = tk.PhotoImage(file='src/gui/button_figs/prebuilt_cnn/resnet152v2.png')
        self.b_resnet152v2 = tk.Button(self.tl_prebuilt_cnn, image=self.p_resnet152v2, command=self.resnet152v2).grid(row=2, column=1)

        self.p_resnext50 = tk.PhotoImage(file='src/gui/button_figs/prebuilt_cnn/resnext50.png')
        self.b_resnext50 = tk.Button(self.tl_prebuilt_cnn, image=self.p_resnext50, command=self.resnext50).grid(row=2, column=2)

        self.p_resnext101 = tk.PhotoImage(file='src/gui/button_figs/prebuilt_cnn/resnext101.png')
        self.b_resnext101 = tk.Button(self.tl_prebuilt_cnn, image=self.p_resnext101, command=self.resnext101).grid(row=2, column=3)

        self.p_inceptionresnetv2 = tk.PhotoImage(file='src/gui/button_figs/prebuilt_cnn/inceptionresnetv2.png')
        self.b_inceptionresnetv2 = tk.Button(self.tl_prebuilt_cnn, image=self.p_inceptionresnetv2, command=self.inceptionresnetv2).grid(row=3, column=0)

        self.p_inceptionv3 = tk.PhotoImage(file='src/gui/button_figs/prebuilt_cnn/inceptionv3.png')
        self.b_inceptionv3 = tk.Button(self.tl_prebuilt_cnn, image=self.p_inceptionv3, command=self.inceptionv3).grid(row=3, column=1)

        self.p_xception = tk.PhotoImage(file='src/gui/button_figs/prebuilt_cnn/xception.png')
        self.b_xception = tk.Button(self.tl_prebuilt_cnn, image=self.p_xception, command=self.xception).grid(row=3, column=2)

        self.p_densenet121 = tk.PhotoImage(file='src/gui/button_figs/prebuilt_cnn/densenet121.png')
        self.b_densenet121 = tk.Button(self.tl_prebuilt_cnn, image=self.p_densenet121, command=self.densenet121).grid(row=3, column=3)

        self.p_densenet169 = tk.PhotoImage(file='src/gui/button_figs/prebuilt_cnn/densenet169.png')
        self.b_densenet169 = tk.Button(self.tl_prebuilt_cnn, image=self.p_densenet169, command=self.densenet169).grid(row=4, column=0)

        self.p_densenet201 = tk.PhotoImage(file='src/gui/button_figs/prebuilt_cnn/densenet201.png')
        self.b_densenet201 = tk.Button(self.tl_prebuilt_cnn, image=self.p_densenet201, command=self.densenet201).grid(row=4, column=1)

        self.p_mobilenet = tk.PhotoImage(file='src/gui/button_figs/prebuilt_cnn/mobilenet.png')
        self.b_mobilenet = tk.Button(self.tl_prebuilt_cnn, image=self.p_mobilenet, command=self.mobilenet).grid(row=4, column=2)

        self.p_mobilenetv2 = tk.PhotoImage(file='src/gui/button_figs/prebuilt_cnn/mobilenetv2.png')
        self.b_mobilenetv2 = tk.Button(self.tl_prebuilt_cnn, image=self.p_mobilenetv2, command=self.mobilenetv2).grid(row=4, column=3)

        self.tl_prebuilt_cnn.resizable(width=False, height=False)
        self.tl_prebuilt_cnn.withdraw()

    def show(self):
        self.tl_prebuilt_cnn.deiconify()

    def lenet5(self):
        self.tl_prebuilt_cnn.withdraw()
        tf.reset_default_graph()
        cwd = os.getcwd()
        config_file = os.path.join(cwd, "prebuilt_configs/lenet5.json")
        configs = load_config(config_file)
        self.controller.set_configs(configs)

    def alexnet(self):
        self.tl_prebuilt_cnn.withdraw()
        tf.reset_default_graph()
        cwd = os.getcwd()
        config_file = os.path.join(cwd, "prebuilt_configs/alexnet.json")
        configs = load_config(config_file)
        self.controller.set_configs(configs)

    def vgg16(self):
        self.tl_prebuilt_cnn.withdraw()
        tf.reset_default_graph()
        cwd = os.getcwd()
        config_file = os.path.join(cwd, "prebuilt_configs/vgg16.json")
        configs = load_config(config_file)
        self.controller.set_configs(configs)

    def vgg19(self):
        self.tl_prebuilt_cnn.withdraw()
        tf.reset_default_graph()
        cwd = os.getcwd()
        config_file = os.path.join(cwd, "prebuilt_configs/vgg19.json")
        configs = load_config(config_file)
        self.controller.set_configs(configs)

    def resnet50(self):
        self.tl_prebuilt_cnn.withdraw()
        tf.reset_default_graph()
        cwd = os.getcwd()
        config_file = os.path.join(cwd, "prebuilt_configs/resnet50.json")
        configs = load_config(config_file)
        self.controller.set_configs(configs)

    def resnet101(self):
        self.tl_prebuilt_cnn.withdraw()
        tf.reset_default_graph()
        cwd = os.getcwd()
        config_file = os.path.join(cwd, "prebuilt_configs/resnet101.json")
        configs = load_config(config_file)
        self.controller.set_configs(configs)

    def resnet152(self):
        self.tl_prebuilt_cnn.withdraw()
        tf.reset_default_graph()
        cwd = os.getcwd()
        config_file = os.path.join(cwd, "prebuilt_configs/resnet152.json")
        configs = load_config(config_file)
        self.controller.set_configs(configs)

    def resnet50v2(self):
        self.tl_prebuilt_cnn.withdraw()
        tf.reset_default_graph()
        cwd = os.getcwd()
        config_file = os.path.join(cwd, "prebuilt_configs/resnet50v2.json")
        configs = load_config(config_file)
        self.controller.set_configs(configs)

    def resnet101v2(self):
        self.tl_prebuilt_cnn.withdraw()
        tf.reset_default_graph()
        cwd = os.getcwd()
        config_file = os.path.join(cwd, "prebuilt_configs/resnet101v2.json")
        configs = load_config(config_file)
        self.controller.set_configs(configs)

    def resnet152v2(self):
        self.tl_prebuilt_cnn.withdraw()
        tf.reset_default_graph()
        cwd = os.getcwd()
        config_file = os.path.join(cwd, "prebuilt_configs/resnet152v2.json")
        configs = load_config(config_file)
        self.controller.set_configs(configs)

    def resnext50(self):
        self.tl_prebuilt_cnn.withdraw()
        tf.reset_default_graph()
        cwd = os.getcwd()
        config_file = os.path.join(cwd, "prebuilt_configs/resnext50.json")
        configs = load_config(config_file)
        self.controller.set_configs(configs)

    def resnext101(self):
        self.tl_prebuilt_cnn.withdraw()
        tf.reset_default_graph()
        cwd = os.getcwd()
        config_file = os.path.join(cwd, "prebuilt_configs/resnext101.json")
        configs = load_config(config_file)
        self.controller.set_configs(configs)

    def inceptionresnetv2(self):
        self.tl_prebuilt_cnn.withdraw()
        tf.reset_default_graph()
        cwd = os.getcwd()
        config_file = os.path.join(cwd, "prebuilt_configs/inceptionresnetv2.json")
        configs = load_config(config_file)
        self.controller.set_configs(configs)

    def inceptionv3(self):
        self.tl_prebuilt_cnn.withdraw()
        tf.reset_default_graph()
        cwd = os.getcwd()
        config_file = os.path.join(cwd, "prebuilt_configs/inceptionv3.json")
        configs = load_config(config_file)
        self.controller.set_configs(configs)

    def xception(self):
        self.tl_prebuilt_cnn.withdraw()
        tf.reset_default_graph()
        cwd = os.getcwd()
        config_file = os.path.join(cwd, "prebuilt_configs/xception.json")
        configs = load_config(config_file)
        self.controller.set_configs(configs)

    def densenet121(self):
        self.tl_prebuilt_cnn.withdraw()
        tf.reset_default_graph()
        cwd = os.getcwd()
        config_file = os.path.join(cwd, "prebuilt_configs/densenet121.json")
        configs = load_config(config_file)
        self.controller.set_configs(configs)

    def densenet169(self):
        self.tl_prebuilt_cnn.withdraw()
        tf.reset_default_graph()
        cwd = os.getcwd()
        config_file = os.path.join(cwd, "prebuilt_configs/densenet169.json")
        configs = load_config(config_file)
        self.controller.set_configs(configs)

    def densenet201(self):
        self.tl_prebuilt_cnn.withdraw()
        tf.reset_default_graph()
        cwd = os.getcwd()
        config_file = os.path.join(cwd, "prebuilt_configs/densenet201.json")
        configs = load_config(config_file)
        self.controller.set_configs(configs)

    def mobilenet(self):
        self.tl_prebuilt_cnn.withdraw()
        tf.reset_default_graph()
        cwd = os.getcwd()
        config_file = os.path.join(cwd, "prebuilt_configs/mobilenet.json")
        configs = load_config(config_file)
        self.controller.set_configs(configs)

    def mobilenetv2(self):
        self.tl_prebuilt_cnn.withdraw()
        tf.reset_default_graph()
        cwd = os.getcwd()
        config_file = os.path.join(cwd, "prebuilt_configs/mobilenetv2.json")
        configs = load_config(config_file)
        self.controller.set_configs(configs)
