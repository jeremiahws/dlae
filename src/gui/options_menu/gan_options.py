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
"""src/gui/options_menu/gan_options.py"""


import tkinter as tk


class GanOptions:
    def __init__(self, controller):
        self.controller = controller
        self.button_heights = 1
        self.button_widths = 15
        self.label_heights = 1
        self.label_widths = 15
        self.entry_widths = 15

        self.tl_gan_options = tk.Toplevel()
        self.tl_gan_options.title('GAN options')
        self.tl_gan_options.resizable(width=False, height=False)
        self.tl_gan_options.wm_protocol('WM_DELETE_WINDOW', self.tl_gan_options.withdraw)
        self.l_patch_gan = tk.Label(self.tl_gan_options, text='Patch GAN:').grid(row=0, column=0, sticky=tk.N+tk.S+tk.E+tk.W)
        self.c_patch_gan = tk.Checkbutton(self.tl_gan_options, variable=self.controller.options_menu.bool_patch_gan).grid(row=0, column=1, sticky=tk.N+tk.S+tk.E+tk.W)
        self.e_patch_gan = tk.Entry(self.tl_gan_options, textvariable=self.controller.options_menu.s_patch_gan).grid(row=0, column=2, sticky=tk.N+tk.S+tk.E+tk.W)

        self.l_label_smooth = tk.Label(self.tl_gan_options, text='Label smoothing:').grid(row=1, column=0, sticky=tk.N+tk.S+tk.E+tk.W)
        self.c_label_smooth = tk.Checkbutton(self.tl_gan_options, variable=self.controller.options_menu.bool_label_smooth).grid(row=1, column=1, sticky=tk.N+tk.S+tk.E+tk.W)
        self.e_label_smooth = tk.Entry(self.tl_gan_options, textvariable=self.controller.options_menu.s_label_smooth).grid(row=1, column=2, sticky=tk.N+tk.S+tk.E+tk.W)

        self.l_label_flip = tk.Label(self.tl_gan_options, text='Label flipping:').grid(row=2, column=0, sticky=tk.N+tk.S+tk.E+tk.W)
        self.c_label_flip = tk.Checkbutton(self.tl_gan_options, variable=self.controller.options_menu.bool_label_flip).grid(row=2, column=1, sticky=tk.N+tk.S+tk.E+tk.W)
        self.e_label_flip = tk.Entry(self.tl_gan_options, textvariable=self.controller.options_menu.s_label_flip).grid(row=2, column=2, sticky=tk.N+tk.S+tk.E+tk.W)

        self.l_balance_train = tk.Label(self.tl_gan_options, text='Balance training:').grid(row=3, column=0, sticky=tk.N+tk.S+tk.E+tk.W)
        self.c_balance_train = tk.Checkbutton(self.tl_gan_options, variable=self.controller.options_menu.bool_balance_train).grid(row=3, column=1, sticky=tk.N+tk.S+tk.E+tk.W)
        self.e_balance_train = tk.Entry(self.tl_gan_options, textvariable=self.controller.options_menu.s_balance_train).grid(row=3, column=2, sticky=tk.N+tk.S+tk.E+tk.W)
        self.tl_gan_options.withdraw()

    def show(self):
        self.tl_gan_options.deiconify()
