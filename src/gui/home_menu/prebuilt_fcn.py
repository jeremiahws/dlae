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
"""src/gui/home_menu/prebuilt_fcn.py"""


import tkinter as tk
import os
from src.utils.general_utils import load_config
import tensorflow as tf


class PrebuiltFullyConvolutionalNetwork:
    def __init__(self, controller):
        self.tl_prebuilt_fcn = tk.Toplevel()
        self.tl_prebuilt_fcn.title('Prebuilt FCNs')
        self.tl_prebuilt_fcn.wm_protocol('WM_DELETE_WINDOW', self.tl_prebuilt_fcn.withdraw)
        self.controller = controller

        self.p_unet2D = tk.PhotoImage(file='src/gui/button_figs/prebuilt_fcn/unet2D.png')
        self.b_unet2D = tk.Button(self.tl_prebuilt_fcn, image=self.p_unet2D, command=self.unet2D).grid(row=0, column=0)

        self.p_unet3D = tk.PhotoImage(file='src/gui/button_figs/prebuilt_fcn/unet3D.png')
        self.b_unet3D = tk.Button(self.tl_prebuilt_fcn, image=self.p_unet3D, command=self.unet3D).grid(row=0, column=1)

        self.p_vgg16_unet = tk.PhotoImage(file='src/gui/button_figs/prebuilt_fcn/vgg16_unet.png')
        self.b_vgg16_unet = tk.Button(self.tl_prebuilt_fcn, image=self.p_vgg16_unet, command=self.vgg16_unet).grid(row=0, column=2)

        self.p_vgg19_unet = tk.PhotoImage(file='src/gui/button_figs/prebuilt_fcn/vgg19_unet.png')
        self.b_vgg19_unet = tk.Button(self.tl_prebuilt_fcn, image=self.p_vgg19_unet, command=self.vgg19_unet).grid(row=0, column=3)

        self.p_densenet121_unet = tk.PhotoImage(file='src/gui/button_figs/prebuilt_fcn/densenet121_unet.png')
        self.b_densenet121_unet = tk.Button(self.tl_prebuilt_fcn, image=self.p_densenet121_unet, command=self.densenet121_unet).grid(row=1, column=0)

        self.p_densenet169_unet = tk.PhotoImage(file='src/gui/button_figs/prebuilt_fcn/densenet169_unet.png')
        self.b_densenet169_unet = tk.Button(self.tl_prebuilt_fcn, image=self.p_densenet169_unet, command=self.densenet169_unet).grid(row=1, column=1)

        self.p_densenet201_unet = tk.PhotoImage(file='src/gui/button_figs/prebuilt_fcn/densenet201_unet.png')
        self.b_densenet201_unet = tk.Button(self.tl_prebuilt_fcn, image=self.p_densenet201_unet, command=self.densenet201_unet).grid(row=1, column=2)

        self.p_xception_unet = tk.PhotoImage(file='src/gui/button_figs/prebuilt_fcn/xception_unet.png')
        self.b_xception_unet = tk.Button(self.tl_prebuilt_fcn, image=self.p_xception_unet, command=self.xception_unet).grid(row=1, column=3)

        self.p_resnet50_unet = tk.PhotoImage(file='src/gui/button_figs/prebuilt_fcn/resnet50_unet.png')
        self.b_resnet50_unet = tk.Button(self.tl_prebuilt_fcn, image=self.p_resnet50_unet, command=self.resnet50_unet).grid(row=2, column=0)

        self.p_resnet101_unet = tk.PhotoImage(file='src/gui/button_figs/prebuilt_fcn/resnet101_unet.png')
        self.b_resnet101_unet = tk.Button(self.tl_prebuilt_fcn, image=self.p_resnet101_unet, command=self.resnet101_unet).grid(row=2, column=1)

        self.p_resnet152_unet = tk.PhotoImage(file='src/gui/button_figs/prebuilt_fcn/resnet152_unet.png')
        self.b_resnet152_unet = tk.Button(self.tl_prebuilt_fcn, image=self.p_resnet152_unet, command=self.resnet152_unet).grid(row=2, column=2)

        self.p_resnet50v2_unet = tk.PhotoImage(file='src/gui/button_figs/prebuilt_fcn/resnet50v2_unet.png')
        self.b_resnet50v2_unet = tk.Button(self.tl_prebuilt_fcn, image=self.p_resnet50v2_unet, command=self.resnet50v2_unet).grid(row=2, column=3)

        self.p_resnet101v2_unet = tk.PhotoImage(file='src/gui/button_figs/prebuilt_fcn/resnet101v2_unet.png')
        self.b_resnet101v2_unet = tk.Button(self.tl_prebuilt_fcn, image=self.p_resnet101v2_unet, command=self.resnet101v2_unet).grid(row=3, column=0)

        self.p_resnet152v2_unet = tk.PhotoImage(file='src/gui/button_figs/prebuilt_fcn/resnet152v2_unet.png')
        self.b_resnet152v2_unet = tk.Button(self.tl_prebuilt_fcn, image=self.p_resnet152v2_unet, command=self.resnet152v2_unet).grid(row=3, column=1)

        self.p_resnext50_unet = tk.PhotoImage(file='src/gui/button_figs/prebuilt_fcn/resnext50_unet.png')
        self.b_resnext50_unet = tk.Button(self.tl_prebuilt_fcn, image=self.p_resnext50_unet, command=self.resnext50_unet).grid(row=3, column=2)

        self.p_resnext101_unet = tk.PhotoImage(file='src/gui/button_figs/prebuilt_fcn/resnext101_unet.png')
        self.b_resnext101_unet = tk.Button(self.tl_prebuilt_fcn, image=self.p_resnext101_unet, command=self.resnext101_unet).grid(row=3, column=3)

        self.p_inceptionv3_unet = tk.PhotoImage(file='src/gui/button_figs/prebuilt_fcn/inceptionv3_unet.png')
        self.b_inceptionv3_unet = tk.Button(self.tl_prebuilt_fcn, image=self.p_inceptionv3_unet, command=self.inceptionv3_unet).grid(row=4, column=0)

        self.p_inceptionresnetv2_unet = tk.PhotoImage(file='src/gui/button_figs/prebuilt_fcn/inceptionresnetv2_unet.png')
        self.b_inceptionresnetv2_unet = tk.Button(self.tl_prebuilt_fcn, image=self.p_inceptionresnetv2_unet, command=self.inceptionresnetv2_unet).grid(row=4, column=1)

        self.p_mobilenet_unet = tk.PhotoImage(file='src/gui/button_figs/prebuilt_fcn/mobilenet_unet.png')
        self.b_mobilenet_unet = tk.Button(self.tl_prebuilt_fcn, image=self.p_mobilenet_unet, command=self.mobilenet_unet).grid(row=4, column=2)

        self.p_mobilenetv2_unet = tk.PhotoImage(file='src/gui/button_figs/prebuilt_fcn/mobilenetv2_unet.png')
        self.b_mobilenetv2_unet = tk.Button(self.tl_prebuilt_fcn, image=self.p_mobilenetv2_unet, command=self.mobilenetv2_unet).grid(row=4, column=3)

        self.tl_prebuilt_fcn.resizable(width=False, height=False)
        self.tl_prebuilt_fcn.withdraw()

    def show(self):
        self.tl_prebuilt_fcn.deiconify()

    def unet2D(self):
        self.tl_prebuilt_fcn.withdraw()
        tf.reset_default_graph()
        cwd = os.getcwd()
        config_file = os.path.join(cwd, "prebuilt_configs/unet2d.json")
        configs = load_config(config_file)
        self.controller.set_configs(configs)

    def unet3D(self):
        self.tl_prebuilt_fcn.withdraw()
        tf.reset_default_graph()
        cwd = os.getcwd()
        config_file = os.path.join(cwd, "prebuilt_configs/unet3d.json")
        configs = load_config(config_file)
        self.controller.set_configs(configs)

    def vgg16_unet(self):
        self.tl_prebuilt_fcn.withdraw()
        tf.reset_default_graph()
        cwd = os.getcwd()
        config_file = os.path.join(cwd, "prebuilt_configs/vgg16_unet.json")
        configs = load_config(config_file)
        self.controller.set_configs(configs)

    def vgg19_unet(self):
        self.tl_prebuilt_fcn.withdraw()
        tf.reset_default_graph()
        cwd = os.getcwd()
        config_file = os.path.join(cwd, "prebuilt_configs/vgg19_unet.json")
        configs = load_config(config_file)
        self.controller.set_configs(configs)

    def densenet121_unet(self):
        self.tl_prebuilt_fcn.withdraw()
        tf.reset_default_graph()
        cwd = os.getcwd()
        config_file = os.path.join(cwd, "prebuilt_configs/densenet121_unet.json")
        configs = load_config(config_file)
        self.controller.set_configs(configs)

    def densenet169_unet(self):
        self.tl_prebuilt_fcn.withdraw()
        tf.reset_default_graph()
        cwd = os.getcwd()
        config_file = os.path.join(cwd, "prebuilt_configs/densenet169_unet.json")
        configs = load_config(config_file)
        self.controller.set_configs(configs)

    def densenet201_unet(self):
        self.tl_prebuilt_fcn.withdraw()
        tf.reset_default_graph()
        cwd = os.getcwd()
        config_file = os.path.join(cwd, "prebuilt_configs/densenet201_unet.json")
        configs = load_config(config_file)
        self.controller.set_configs(configs)

    def xception_unet(self):
        self.tl_prebuilt_fcn.withdraw()
        tf.reset_default_graph()
        cwd = os.getcwd()
        config_file = os.path.join(cwd, "prebuilt_configs/xception_unet.json")
        configs = load_config(config_file)
        self.controller.set_configs(configs)

    def resnet50_unet(self):
        self.tl_prebuilt_fcn.withdraw()
        tf.reset_default_graph()
        cwd = os.getcwd()
        config_file = os.path.join(cwd, "prebuilt_configs/resnet50_unet.json")
        configs = load_config(config_file)
        self.controller.set_configs(configs)

    def resnet101_unet(self):
        self.tl_prebuilt_fcn.withdraw()
        tf.reset_default_graph()
        cwd = os.getcwd()
        config_file = os.path.join(cwd, "prebuilt_configs/resnet101_unet.json")
        configs = load_config(config_file)
        self.controller.set_configs(configs)

    def resnet152_unet(self):
        self.tl_prebuilt_fcn.withdraw()
        tf.reset_default_graph()
        cwd = os.getcwd()
        config_file = os.path.join(cwd, "prebuilt_configs/resnet152_unet.json")
        configs = load_config(config_file)
        self.controller.set_configs(configs)

    def resnet50v2_unet(self):
        self.tl_prebuilt_fcn.withdraw()
        tf.reset_default_graph()
        cwd = os.getcwd()
        config_file = os.path.join(cwd, "prebuilt_configs/resnet50v2_unet.json")
        configs = load_config(config_file)
        self.controller.set_configs(configs)

    def resnet101v2_unet(self):
        self.tl_prebuilt_fcn.withdraw()
        tf.reset_default_graph()
        cwd = os.getcwd()
        config_file = os.path.join(cwd, "prebuilt_configs/resnet101v2_unet.json")
        configs = load_config(config_file)
        self.controller.set_configs(configs)

    def resnet152v2_unet(self):
        self.tl_prebuilt_fcn.withdraw()
        tf.reset_default_graph()
        cwd = os.getcwd()
        config_file = os.path.join(cwd, "prebuilt_configs/resnet152v2_unet.json")
        configs = load_config(config_file)
        self.controller.set_configs(configs)

    def resnext50_unet(self):
        self.tl_prebuilt_fcn.withdraw()
        tf.reset_default_graph()
        cwd = os.getcwd()
        config_file = os.path.join(cwd, "prebuilt_configs/resnext50_unet.json")
        configs = load_config(config_file)
        self.controller.set_configs(configs)

    def resnext101_unet(self):
        self.tl_prebuilt_fcn.withdraw()
        tf.reset_default_graph()
        cwd = os.getcwd()
        config_file = os.path.join(cwd, "prebuilt_configs/resnext101_unet.json")
        configs = load_config(config_file)
        self.controller.set_configs(configs)

    def inceptionv3_unet(self):
        self.tl_prebuilt_fcn.withdraw()
        tf.reset_default_graph()
        cwd = os.getcwd()
        config_file = os.path.join(cwd, "prebuilt_configs/inceptionv3_unet.json")
        configs = load_config(config_file)
        self.controller.set_configs(configs)

    def inceptionresnetv2_unet(self):
        self.tl_prebuilt_fcn.withdraw()
        tf.reset_default_graph()
        cwd = os.getcwd()
        config_file = os.path.join(cwd, "prebuilt_configs/inceptionresnetv2_unet.json")
        configs = load_config(config_file)
        self.controller.set_configs(configs)

    def mobilenet_unet(self):
        self.tl_prebuilt_fcn.withdraw()
        tf.reset_default_graph()
        cwd = os.getcwd()
        config_file = os.path.join(cwd, "prebuilt_configs/mobilenet_unet.json")
        configs = load_config(config_file)
        self.controller.set_configs(configs)

    def mobilenetv2_unet(self):
        self.tl_prebuilt_fcn.withdraw()
        tf.reset_default_graph()
        cwd = os.getcwd()
        config_file = os.path.join(cwd, "prebuilt_configs/mobilenetv2_unet.json")
        configs = load_config(config_file)
        self.controller.set_configs(configs)
