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
"""src/gui/file_menu/config_file.py"""


import tkinter as tk


class ConfigurationFile:
    def __init__(self, controller):
        self.controller = controller
        self.button_heights = 1
        self.button_widths = 15
        self.label_heights = 1
        self.label_widths = 15
        self.entry_widths = 15

        self.tl_config_def = tk.Toplevel()
        self.tl_config_def.title('Config file type...')
        self.tl_config_def.resizable(width=False, height=False)
        self.tl_config_def.wm_protocol('WM_DELETE_WINDOW', self.tl_config_def.withdraw)

        self.om_model_signal = tk.OptionMenu(self.tl_config_def, self.controller.file_menu.s_model_signal, *self.controller.file_menu.o_model_signal)
        self.om_model_signal.config()
        self.om_model_signal.grid(row=0, column=0, sticky=tk.N+tk.S+tk.E+tk.W)

        self.om_type_signal = tk.OptionMenu(self.tl_config_def, self.controller.file_menu.s_type_signal, *self.controller.file_menu.o_type_signal)
        self.om_type_signal.config()
        self.om_type_signal.grid(row=0, column=1, sticky=tk.N+tk.S+tk.E+tk.W)

        self.l_input_shape = tk.Label(self.tl_config_def, text='Input shape:').grid(row=1, column=0, sticky=tk.N+tk.S+tk.E+tk.W)
        self.e_input_shape = tk.Entry(self.tl_config_def, textvariable=self.controller.file_menu.s_input_shape).grid(row=1, column=1)
        self.tl_config_def.withdraw()

    def show(self):
        self.tl_config_def.deiconify()
