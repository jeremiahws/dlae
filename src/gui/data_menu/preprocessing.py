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
"""src/gui/data_menu/preprocessing.py"""


import tkinter as tk


class Preprocessing:
    def __init__(self, controller):
        """
        Constructor class to build the preprocessing pop-out menu for the
        data menu of the graphical user interface.
        :param controller: the GUI variable controller

        Attributes:
            tl_preprocessing -- pop-out menu for postprocessing steps
            l_data_ranges -- label for defining the range(s) of values in the images
            e_data_min -- entry for to define the minimum value(s) in the images
            e_data_max -- entry for to define the maximum value(s) in the images
            l_image_context -- label for defining the image context to be inferred from (2D, 3D)
            om_image_context -- option menu for selecting the image context to be inferred from (2D, 3D)
            l_normalization_type -- label for defining the image normalization type
            om_normalization_type -- option menu for defining the image normalization type
            l_to_categorical --
            c_to_categorical --
            l_weight_loss --
            c_weight_loss --
            l_reshapeX --
            c_reshapeX --
            e_reshapeX --
            l_permuteX --
            c_permuteX --
            e_permuteX --
            l_repeatX --
            c_repeatX --
            e_repeatX --
            l_reshapeY --
            c_reshapeY --
            e_reshapeY --
            l_permuteY --
            c_permuteY --
            e_permuteY --
        """
        self.tl_preprocessing = tk.Toplevel()
        self.tl_preprocessing.title('Preprocessing steps')
        self.tl_preprocessing.resizable(width=False, height=False)
        self.tl_preprocessing.wm_protocol('WM_DELETE_WINDOW', self.tl_preprocessing.withdraw)

        self.l_data_ranges = tk.Label(self.tl_preprocessing, text='Data minimum, maximum:').grid(row=0, column=0, sticky=tk.N+tk.S+tk.E+tk.W)
        self.e_data_min = tk.Entry(self.tl_preprocessing, textvariable=controller.data_menu.s_data_min).grid(row=0, column=1, sticky=tk.N+tk.S+tk.E+tk.W)
        self.e_data_max = tk.Entry(self.tl_preprocessing, textvariable=controller.data_menu.s_data_max).grid(row=0, column=2, sticky=tk.N+tk.S+tk.E+tk.W)

        self.l_image_context = tk.Label(self.tl_preprocessing, text='Image context:').grid(row=1, column=0, sticky=tk.N+tk.S+tk.E+tk.W)
        self.om_image_context = tk.OptionMenu(self.tl_preprocessing, controller.data_menu.s_image_context, *controller.data_menu.o_image_context)
        self.om_image_context.config()
        self.om_image_context.grid(row=1, column=1, sticky=tk.N+tk.S+tk.E+tk.W)

        self.l_normalization_type = tk.Label(self.tl_preprocessing, text='Normalization type:').grid(row=2, column=0, sticky=tk.N+tk.S+tk.E+tk.W)
        self.om_normalization_type = tk.OptionMenu(self.tl_preprocessing, controller.data_menu.s_normalization_type, *controller.data_menu.o_normalization_type)
        self.om_normalization_type.config()
        self.om_normalization_type.grid(row=2, column=1, sticky=tk.N+tk.S+tk.E+tk.W)

        self.l_to_categorical = tk.Label(self.tl_preprocessing, text='Convert y to categorical:').grid(row=3, column=0, sticky=tk.N+tk.S+tk.E+tk.W)
        self.c_to_categorical = tk.Checkbutton(self.tl_preprocessing, variable=controller.data_menu.bool_to_categorical).grid(row=3, column=1, sticky=tk.N+tk.S+tk.E+tk.W)

        self.l_weight_loss = tk.Label(self.tl_preprocessing, text='Weight loss function:').grid(row=4, column=0, sticky=tk.N+tk.S+tk.E+tk.W)
        self.c_weight_loss = tk.Checkbutton(self.tl_preprocessing, variable=controller.data_menu.bool_weight_loss).grid(row=4, column=1, sticky=tk.N+tk.S+tk.E+tk.W)

        self.l_reshapeX = tk.Label(self.tl_preprocessing, text='Reshape X:').grid(row=5, column=0, sticky=tk.N+tk.S+tk.E+tk.W)
        self.c_reshapeX = tk.Checkbutton(self.tl_preprocessing, variable=controller.data_menu.bool_reshapeX).grid(row=5, column=1, sticky=tk.N+tk.S+tk.E+tk.W)
        self.e_reshapeX = tk.Entry(self.tl_preprocessing, textvariable=controller.data_menu.s_reshapeX).grid(row=5, column=2, sticky=tk.N+tk.S+tk.E+tk.W)

        self.l_permuteX = tk.Label(self.tl_preprocessing, text='Permute X:').grid(row=6, column=0, sticky=tk.N+tk.S+tk.E+tk.W)
        self.c_permuteX = tk.Checkbutton(self.tl_preprocessing, variable=controller.data_menu.bool_permuteX).grid(row=6, column=1, sticky=tk.N+tk.S+tk.E+tk.W)
        self.e_permuteX = tk.Entry(self.tl_preprocessing, textvariable=controller.data_menu.s_permuteX).grid(row=6, column=2, sticky=tk.N+tk.S+tk.E+tk.W)

        self.l_repeatX = tk.Label(self.tl_preprocessing, text='Repeat X along channels:').grid(row=7, column=0, sticky=tk.N+tk.S+tk.E+tk.W)
        self.c_repeatX = tk.Checkbutton(self.tl_preprocessing, variable=controller.data_menu.bool_repeatX).grid(row=7, column=1, sticky=tk.N+tk.S+tk.E+tk.W)
        self.e_repeatX = tk.Entry(self.tl_preprocessing, textvariable=controller.data_menu.s_repeatX).grid(row=7, column=2, sticky=tk.N+tk.S+tk.E+tk.W)

        self.l_reshapeY = tk.Label(self.tl_preprocessing, text='Reshape y:').grid(row=8, column=0, sticky=tk.N+tk.S+tk.E+tk.W)
        self.c_reshapeY = tk.Checkbutton(self.tl_preprocessing, variable=controller.data_menu.bool_reshapeY).grid(row=8, column=1, sticky=tk.N+tk.S+tk.E+tk.W)
        self.e_reshapeY = tk.Entry(self.tl_preprocessing, textvariable=controller.data_menu.s_reshapeY).grid(row=8, column=2, sticky=tk.N+tk.S+tk.E+tk.W)

        self.l_permuteY = tk.Label(self.tl_preprocessing, text='Permute y:').grid(row=9, column=0, sticky=tk.N+tk.S+tk.E+tk.W)
        self.c_permuteY = tk.Checkbutton(self.tl_preprocessing, variable=controller.data_menu.bool_permuteY).grid(row=9, column=1, sticky=tk.N+tk.S+tk.E+tk.W)
        self.e_permuteY = tk.Entry(self.tl_preprocessing, textvariable=controller.data_menu.s_permuteY).grid(row=9, column=2, sticky=tk.N+tk.S+tk.E+tk.W)
        self.tl_preprocessing.withdraw()

    def show(self):
        self.tl_preprocessing.deiconify()
