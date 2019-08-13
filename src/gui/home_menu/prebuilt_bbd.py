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
"""src/gui/home_menu/prebuilt_bbd.py"""


import tkinter as tk
from src.utils.general_utils import load_config
import os
import tensorflow as tf


class PrebuiltBoundingBoxDetector:
    def __init__(self, controller):
        self.tl_prebuilt_bbd = tk.Toplevel()
        self.tl_prebuilt_bbd.title('Prebuilt BBDs')
        self.tl_prebuilt_bbd.wm_protocol('WM_DELETE_WINDOW', self.tl_prebuilt_bbd.withdraw)
        self.controller = controller

        self.p_ssd = tk.PhotoImage(file='src/gui/button_figs/prebuilt_bbd/ssd.png')
        self.b_ssd = tk.Button(self.tl_prebuilt_bbd, image=self.p_ssd, command=self.ssd).grid(row=0, column=0)

        self.p_vgg16_ssd = tk.PhotoImage(file='src/gui/button_figs/prebuilt_bbd/vgg16_ssd.png')
        self.b_vgg16_ssd = tk.Button(self.tl_prebuilt_bbd, image=self.p_vgg16_ssd, command=self.vgg16_ssd).grid(row=0, column=1)

        self.p_vgg19_ssd = tk.PhotoImage(file='src/gui/button_figs/prebuilt_bbd/vgg19_ssd.png')
        self.b_vgg19_ssd = tk.Button(self.tl_prebuilt_bbd, image=self.p_vgg19_ssd, command=self.vgg19_ssd).grid(row=0, column=2)

        self.p_densenet121_ssd = tk.PhotoImage(file='src/gui/button_figs/prebuilt_bbd/densenet121_ssd.png')
        self.b_densenet121_ssd = tk.Button(self.tl_prebuilt_bbd, image=self.p_densenet121_ssd, command=self.densenet121_ssd).grid(row=0, column=3)

        self.p_densenet169_ssd = tk.PhotoImage(file='src/gui/button_figs/prebuilt_bbd/densenet169_ssd.png')
        self.b_densenet169_ssd = tk.Button(self.tl_prebuilt_bbd, image=self.p_densenet169_ssd, command=self.densenet169_ssd).grid(row=1, column=0)

        self.p_densenet201_ssd = tk.PhotoImage(file='src/gui/button_figs/prebuilt_bbd/densenet201_ssd.png')
        self.b_densenet201_ssd = tk.Button(self.tl_prebuilt_bbd, image=self.p_densenet201_ssd, command=self.densenet201_ssd).grid(row=1, column=1)

        self.p_xception_ssd = tk.PhotoImage(file='src/gui/button_figs/prebuilt_bbd/xception_ssd.png')
        self.b_xception_ssd = tk.Button(self.tl_prebuilt_bbd, image=self.p_xception_ssd, command=self.xception_ssd).grid(row=1, column=2)

        self.p_resnet50_ssd = tk.PhotoImage(file='src/gui/button_figs/prebuilt_bbd/resnet50_ssd.png')
        self.b_resnet50_ssd = tk.Button(self.tl_prebuilt_bbd, image=self.p_resnet50_ssd, command=self.resnet50_ssd).grid(row=1, column=3)

        self.p_resnet101_ssd = tk.PhotoImage(file='src/gui/button_figs/prebuilt_bbd/resnet101_ssd.png')
        self.b_resnet101_ssd = tk.Button(self.tl_prebuilt_bbd, image=self.p_resnet101_ssd, command=self.resnet101_ssd).grid(row=2, column=0)

        self.p_resnet152_ssd = tk.PhotoImage(file='src/gui/button_figs/prebuilt_bbd/resnet152_ssd.png')
        self.b_resnet152_ssd = tk.Button(self.tl_prebuilt_bbd, image=self.p_resnet152_ssd, command=self.resnet152_ssd).grid(row=2, column=1)

        self.p_resnet50v2_ssd = tk.PhotoImage(file='src/gui/button_figs/prebuilt_bbd/resnet50v2_ssd.png')
        self.b_resnet50v2_ssd = tk.Button(self.tl_prebuilt_bbd, image=self.p_resnet50v2_ssd, command=self.resnet50v2_ssd).grid(row=2, column=2)

        self.p_resnet101v2_ssd = tk.PhotoImage(file='src/gui/button_figs/prebuilt_bbd/resnet101v2_ssd.png')
        self.b_resnet101v2_ssd = tk.Button(self.tl_prebuilt_bbd, image=self.p_resnet101v2_ssd, command=self.resnet101v2_ssd).grid(row=2, column=3)

        self.p_resnet152v2_ssd = tk.PhotoImage(file='src/gui/button_figs/prebuilt_bbd/resnet152v2_ssd.png')
        self.b_resnet152v2_ssd = tk.Button(self.tl_prebuilt_bbd, image=self.p_resnet152v2_ssd, command=self.resnet152v2_ssd).grid(row=3, column=0)

        self.p_resnext50_ssd = tk.PhotoImage(file='src/gui/button_figs/prebuilt_bbd/resnext50_ssd.png')
        self.b_resnext50_ssd = tk.Button(self.tl_prebuilt_bbd, image=self.p_resnext50_ssd, command=self.resnext50_ssd).grid(row=3, column=1)

        self.p_resnext101_ssd = tk.PhotoImage(file='src/gui/button_figs/prebuilt_bbd/resnext101_ssd.png')
        self.b_resnext101_ssd = tk.Button(self.tl_prebuilt_bbd, image=self.p_resnext101_ssd, command=self.resnext101_ssd).grid(row=3, column=2)

        self.p_inceptionv3_ssd = tk.PhotoImage(file='src/gui/button_figs/prebuilt_bbd/inceptionv3_ssd.png')
        self.b_inceptionv3_ssd = tk.Button(self.tl_prebuilt_bbd, image=self.p_inceptionv3_ssd, command=self.inceptionv3_ssd).grid(row=3, column=3)

        self.p_inceptionresnetv2_ssd = tk.PhotoImage(file='src/gui/button_figs/prebuilt_bbd/inceptionresnetv2_ssd.png')
        self.b_inceptionresnetv2_ssd = tk.Button(self.tl_prebuilt_bbd, image=self.p_inceptionresnetv2_ssd, command=self.inceptionresnetv2_ssd).grid(row=4, column=0)

        self.p_mobilenet_ssd = tk.PhotoImage(file='src/gui/button_figs/prebuilt_bbd/mobilenet_ssd.png')
        self.b_mobilenet_ssd = tk.Button(self.tl_prebuilt_bbd, image=self.p_mobilenet_ssd, command=self.mobilenet_ssd).grid(row=4, column=1)

        self.p_mobilenetv2_ssd = tk.PhotoImage(file='src/gui/button_figs/prebuilt_bbd/mobilenetv2_ssd.png')
        self.b_mobilenetv2_ssd = tk.Button(self.tl_prebuilt_bbd, image=self.p_mobilenetv2_ssd, command=self.mobilenetv2_ssd).grid(row=4, column=2)

        self.tl_prebuilt_bbd.resizable(width=False, height=False)
        self.tl_prebuilt_bbd.withdraw()

    def show(self):
        self.tl_prebuilt_bbd.deiconify()

    def ssd(self):
        self.tl_prebuilt_bbd.withdraw()
        tf.reset_default_graph()
        cwd = os.getcwd()
        config_file = os.path.join(cwd, "prebuilt_configs/ssd.json")
        configs = load_config(config_file)
        self.controller.set_configs(configs)

    def vgg16_ssd(self):
        self.tl_prebuilt_bbd.withdraw()
        tf.reset_default_graph()
        cwd = os.getcwd()
        config_file = os.path.join(cwd, "prebuilt_configs/vgg16_ssd.json")
        configs = load_config(config_file)
        self.controller.set_configs(configs)

    def vgg19_ssd(self):
        self.tl_prebuilt_bbd.withdraw()
        tf.reset_default_graph()
        cwd = os.getcwd()
        config_file = os.path.join(cwd, "prebuilt_configs/vgg19_ssd.json")
        configs = load_config(config_file)
        self.controller.set_configs(configs)

    def densenet121_ssd(self):
        self.tl_prebuilt_bbd.withdraw()
        tf.reset_default_graph()
        cwd = os.getcwd()
        config_file = os.path.join(cwd, "prebuilt_configs/densenet121_ssd.json")
        configs = load_config(config_file)
        self.controller.set_configs(configs)

    def densenet169_ssd(self):
        self.tl_prebuilt_bbd.withdraw()
        tf.reset_default_graph()
        cwd = os.getcwd()
        config_file = os.path.join(cwd, "prebuilt_configs/densenet169_ssd.json")
        configs = load_config(config_file)
        self.controller.set_configs(configs)

    def densenet201_ssd(self):
        self.tl_prebuilt_bbd.withdraw()
        tf.reset_default_graph()
        cwd = os.getcwd()
        config_file = os.path.join(cwd, "prebuilt_configs/densenet201_ssd.json")
        configs = load_config(config_file)
        self.controller.set_configs(configs)

    def xception_ssd(self):
        self.tl_prebuilt_bbd.withdraw()
        tf.reset_default_graph()
        cwd = os.getcwd()
        config_file = os.path.join(cwd, "prebuilt_configs/xception_ssd.json")
        configs = load_config(config_file)
        self.controller.set_configs(configs)

    def resnet50_ssd(self):
        self.tl_prebuilt_bbd.withdraw()
        tf.reset_default_graph()
        cwd = os.getcwd()
        config_file = os.path.join(cwd, "prebuilt_configs/resnet50_ssd.json")
        configs = load_config(config_file)
        self.controller.set_configs(configs)

    def resnet101_ssd(self):
        self.tl_prebuilt_bbd.withdraw()
        tf.reset_default_graph()
        cwd = os.getcwd()
        config_file = os.path.join(cwd, "prebuilt_configs/resnet101_ssd.json")
        configs = load_config(config_file)
        self.controller.set_configs(configs)

    def resnet152_ssd(self):
        self.tl_prebuilt_bbd.withdraw()
        tf.reset_default_graph()
        cwd = os.getcwd()
        config_file = os.path.join(cwd, "prebuilt_configs/resnet152_ssd.json")
        configs = load_config(config_file)
        self.controller.set_configs(configs)

    def resnet50v2_ssd(self):
        self.tl_prebuilt_bbd.withdraw()
        tf.reset_default_graph()
        cwd = os.getcwd()
        config_file = os.path.join(cwd, "prebuilt_configs/resnet50v2_ssd.json")
        configs = load_config(config_file)
        self.controller.set_configs(configs)

    def resnet101v2_ssd(self):
        self.tl_prebuilt_bbd.withdraw()
        tf.reset_default_graph()
        cwd = os.getcwd()
        config_file = os.path.join(cwd, "prebuilt_configs/resnet101v2_ssd.json")
        configs = load_config(config_file)
        self.controller.set_configs(configs)

    def resnet152v2_ssd(self):
        self.tl_prebuilt_bbd.withdraw()
        tf.reset_default_graph()
        cwd = os.getcwd()
        config_file = os.path.join(cwd, "prebuilt_configs/resnet152v2_ssd.json")
        configs = load_config(config_file)
        self.controller.set_configs(configs)

    def resnext50_ssd(self):
        self.tl_prebuilt_bbd.withdraw()
        tf.reset_default_graph()
        cwd = os.getcwd()
        config_file = os.path.join(cwd, "prebuilt_configs/resnext50_ssd.json")
        configs = load_config(config_file)
        self.controller.set_configs(configs)

    def resnext101_ssd(self):
        self.tl_prebuilt_bbd.withdraw()
        tf.reset_default_graph()
        cwd = os.getcwd()
        config_file = os.path.join(cwd, "prebuilt_configs/resnext101_ssd.json")
        configs = load_config(config_file)
        self.controller.set_configs(configs)

    def inceptionv3_ssd(self):
        self.tl_prebuilt_bbd.withdraw()
        tf.reset_default_graph()
        cwd = os.getcwd()
        config_file = os.path.join(cwd, "prebuilt_configs/inceptionv3_ssd.json")
        configs = load_config(config_file)
        self.controller.set_configs(configs)

    def inceptionresnetv2_ssd(self):
        self.tl_prebuilt_bbd.withdraw()
        tf.reset_default_graph()
        cwd = os.getcwd()
        config_file = os.path.join(cwd, "prebuilt_configs/inceptionresnetv2_ssd.json")
        configs = load_config(config_file)
        self.controller.set_configs(configs)

    def mobilenet_ssd(self):
        self.tl_prebuilt_bbd.withdraw()
        tf.reset_default_graph()
        cwd = os.getcwd()
        config_file = os.path.join(cwd, "prebuilt_configs/mobilenet_ssd.json")
        configs = load_config(config_file)
        self.controller.set_configs(configs)

    def mobilenetv2_ssd(self):
        self.tl_prebuilt_bbd.withdraw()
        tf.reset_default_graph()
        cwd = os.getcwd()
        config_file = os.path.join(cwd, "prebuilt_configs/mobilenetv2_ssd.json")
        configs = load_config(config_file)
        self.controller.set_configs(configs)
