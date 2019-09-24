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
        self.e_to_categorical = tk.Entry(self.tl_preprocessing, textvariable=controller.data_menu.s_num_categories).grid(row=3, column=2, sticky=tk.N+tk.S+tk.E+tk.W)

        self.l_weight_loss = tk.Label(self.tl_preprocessing, text='Weight loss function:').grid(row=4, column=0, sticky=tk.N+tk.S+tk.E+tk.W)
        self.c_weight_loss = tk.Checkbutton(self.tl_preprocessing, variable=controller.data_menu.bool_weight_loss).grid(row=4, column=1, sticky=tk.N+tk.S+tk.E+tk.W)

        self.l_repeatX = tk.Label(self.tl_preprocessing, text='Repeat X along channels:').grid(row=5, column=0, sticky=tk.N+tk.S+tk.E+tk.W)
        self.c_repeatX = tk.Checkbutton(self.tl_preprocessing, variable=controller.data_menu.bool_repeatX).grid(row=5, column=1, sticky=tk.N+tk.S+tk.E+tk.W)
        self.e_repeatX = tk.Entry(self.tl_preprocessing, textvariable=controller.data_menu.s_repeatX).grid(row=5, column=2, sticky=tk.N+tk.S+tk.E+tk.W)

        self.tl_preprocessing.withdraw()

    def show(self):
        self.tl_preprocessing.deiconify()
