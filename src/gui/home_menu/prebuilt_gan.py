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
"""src/gui/home_menu/prebuilt_gan.py"""


import tkinter as tk
from src.utils.general_utils import load_config
import os
import tensorflow as tf


class PrebuiltGenerativeAdversarialNetwork:
    def __init__(self, controller):
        self.tl_prebuilt_gan = tk.Toplevel()
        self.tl_prebuilt_gan.title('Prebuilt GANs')
        self.tl_prebuilt_gan.wm_protocol('WM_DELETE_WINDOW', self.tl_prebuilt_gan.withdraw)
        self.controller = controller

        self.p_pix2pix = tk.PhotoImage(file='src/gui/button_figs/prebuilt_gan/pix2pix.png')
        self.b_pix2pix = tk.Button(self.tl_prebuilt_gan, image=self.p_pix2pix, command=self.pix2pix).grid(row=0, column=0)

        self.p_cyclegan = tk.PhotoImage(file='src/gui/button_figs/prebuilt_gan/cyclegan.png')
        self.b_cyclegan = tk.Button(self.tl_prebuilt_gan, image=self.p_cyclegan, command=self.cyclegan).grid(row=0, column=1)

        self.tl_prebuilt_gan.resizable(width=False, height=False)
        self.tl_prebuilt_gan.withdraw()

    def show(self):
        self.tl_prebuilt_gan.deiconify()

    def pix2pix(self):
        self.tl_prebuilt_gan.withdraw()
        tf.reset_default_graph()
        cwd = os.getcwd()
        config_file = os.path.join(cwd, "prebuilt_configs/pix2pix.json")
        configs = load_config(config_file)
        self.controller.set_configs(configs)

    def cyclegan(self):
        self.tl_prebuilt_gan.withdraw()
        tf.reset_default_graph()
        cwd = os.getcwd()
        config_file = os.path.join(cwd, "prebuilt_configs/cyclegan.json")
        configs = load_config(config_file)
        self.controller.set_configs(configs)
