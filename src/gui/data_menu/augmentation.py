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
"""src/gui/data_menu/augmentation.py"""


import tkinter as tk


class Augmentation:
    def __init__(self, controller):
        self.tl_augmentation = tk.Toplevel()
        self.tl_augmentation.title('Augmentation')
        self.tl_augmentation.resizable(width=False, height=False)
        self.tl_augmentation.wm_protocol('WM_DELETE_WINDOW', self.tl_augmentation.withdraw)

        self.l_augmentation = tk.Label(self.tl_augmentation, text='Apply augmentation:').grid(row=0, column=0, sticky=tk.N+tk.S+tk.E+tk.W)
        self.c_augmentation = tk.Checkbutton(self.tl_augmentation, variable=controller.data_menu.bool_augmentation).grid(row=0, column=1, sticky=tk.N+tk.S+tk.E+tk.W)

        self.l_centering = tk.Label(self.tl_augmentation, text='Centering featurewise, samplewise:').grid(row=1, column=0, sticky=tk.N+tk.S+tk.E+tk.W)
        self.c_fw_centering = tk.Checkbutton(self.tl_augmentation, variable=controller.data_menu.bool_fw_centering).grid(row=1, column=1, sticky=tk.N+tk.S+tk.E+tk.W)
        self.c_sw_centering = tk.Checkbutton(self.tl_augmentation, variable=controller.data_menu.bool_sw_centering).grid(row=1, column=2, sticky=tk.N+tk.S+tk.E+tk.W)

        self.l_normalization = tk.Label(self.tl_augmentation, text='Normalization featurewise, samplewise:').grid(row=2, column=0, sticky=tk.N+tk.S+tk.E+tk.W)
        self.c_fw_normalization = tk.Checkbutton(self.tl_augmentation, variable=controller.data_menu.bool_fw_normalization).grid(row=2, column=1, sticky=tk.N+tk.S+tk.E+tk.W)
        self.c_sw_normalization = tk.Checkbutton(self.tl_augmentation, variable=controller.data_menu.bool_sw_normalization).grid(row=2, column=2, sticky=tk.N+tk.S+tk.E+tk.W)

        self.l_shift_ranges = tk.Label(self.tl_augmentation, text='Width, height shift range:').grid(row=3, column=0, sticky=tk.N+tk.S+tk.E+tk.W)
        self.e_width_range = tk.Entry(self.tl_augmentation, textvariable=controller.data_menu.s_width_shift).grid(row=3, column=1, sticky=tk.N+tk.S+tk.E+tk.W)
        self.e_height_range = tk.Entry(self.tl_augmentation, textvariable=controller.data_menu.s_height_shift).grid(row=3, column=2, sticky=tk.N+tk.S+tk.E+tk.W)

        self.l_rotation_range = tk.Label(self.tl_augmentation, text='Rotation range:').grid(row=4, column=0, sticky=tk.N+tk.S+tk.E+tk.W)
        self.e_rotation_range = tk.Entry(self.tl_augmentation, textvariable=controller.data_menu.s_rotation_range).grid(row=4, column=1, sticky=tk.N+tk.S+tk.E+tk.W)

        self.l_brightness_range = tk.Label(self.tl_augmentation, text='Brightness range:').grid(row=5, column=0, sticky=tk.N+tk.S+tk.E+tk.W)
        self.e_brightness_range = tk.Entry(self.tl_augmentation, textvariable=controller.data_menu.s_brightness_range).grid(row=5, column=1, sticky=tk.N+tk.S+tk.E+tk.W)

        self.l_shear_range = tk.Label(self.tl_augmentation, text='Shear range:').grid(row=6, column=0, sticky=tk.N+tk.S+tk.E+tk.W)
        self.e_shear_range = tk.Entry(self.tl_augmentation, textvariable=controller.data_menu.s_shear_range).grid(row=6, column=1, sticky=tk.N+tk.S+tk.E+tk.W)

        self.l_zoom_range = tk.Label(self.tl_augmentation, text='Zoom range:').grid(row=7, column=0, sticky=tk.N+tk.S+tk.E+tk.W)
        self.e_zoom_range = tk.Entry(self.tl_augmentation, textvariable=controller.data_menu.s_zoom_range).grid(row=7, column=1, sticky=tk.N+tk.S+tk.E+tk.W)

        self.l_channel_shift_range = tk.Label(self.tl_augmentation, text='Channel shift range:').grid(row=8, column=0, sticky=tk.N+tk.S+tk.E+tk.W)
        self.e_channel_shift_range = tk.Entry(self.tl_augmentation, textvariable=controller.data_menu.s_channel_shift_range).grid(row=8, column=1, sticky=tk.N+tk.S+tk.E+tk.W)

        self.l_fill_mode = tk.Label(self.tl_augmentation, text='Fill mode:').grid(row=9, column=0, sticky=tk.N+tk.S+tk.E+tk.W)
        self.om_fill_mode = tk.OptionMenu(self.tl_augmentation, controller.data_menu.s_fill_mode, *controller.data_menu.o_fill_mode)
        self.om_fill_mode.config()
        self.om_fill_mode.grid(row=9, column=1, sticky=tk.N+tk.S+tk.E+tk.W)
        self.e_fill_mode = tk.Entry(self.tl_augmentation, textvariable=controller.data_menu.s_cval).grid(row=9, column=2, sticky=tk.N+tk.S+tk.E+tk.W)

        self.l_flips = tk.Label(self.tl_augmentation, text='Horizontal, vertical flips:').grid(row=10, column=0, sticky=tk.N+tk.S+tk.E+tk.W)
        self.c_horizontal_flip = tk.Checkbutton(self.tl_augmentation, variable=controller.data_menu.bool_horizontal_flip).grid(row=10, column=1, sticky=tk.N+tk.S+tk.E+tk.W)
        self.c_vertical_flip = tk.Checkbutton(self.tl_augmentation, variable=controller.data_menu.bool_vertical_flip).grid(row=10, column=2, sticky=tk.N+tk.S+tk.E+tk.W)

        self.tl_augmentation.withdraw()

    def show(self):
        self.tl_augmentation.deiconify()
