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
"""src/gui/options_menu/save_configs.py"""


import tkinter as tk


class SaveConfigurations:
    def __init__(self, controller):
        self.controller = controller
        self.button_heights = 1
        self.button_widths = 15
        self.label_heights = 1
        self.label_widths = 15
        self.entry_widths = 15

        self.tl_save_configs = tk.Toplevel()
        self.tl_save_configs.title('Save configurations')
        self.tl_save_configs.resizable(width=False, height=False)
        self.tl_save_configs.wm_protocol('WM_DELETE_WINDOW', self.tl_save_configs.withdraw)
        self.l_save_model = tk.Label(self.tl_save_configs, text='Save model:').grid(row=0, column=0, sticky=tk.N+tk.S+tk.E+tk.W)
        self.c_save_model = tk.Checkbutton(self.tl_save_configs, variable=self.controller.options_menu.bool_save_model).grid(row=0, column=1, sticky=tk.N+tk.S+tk.E+tk.W)

        self.l_save_model_path = tk.Label(self.tl_save_configs, text='Model path:').grid(row=0, column=2, sticky=tk.N+tk.S+tk.E+tk.W)
        self.e_save_model_path = tk.Entry(self.tl_save_configs, textvariable=self.controller.options_menu.s_save_model_path).grid(row=0, column=3, sticky=tk.N+tk.S+tk.E+tk.W)

        self.l_save_csv = tk.Label(self.tl_save_configs, text='Save csv:').grid(row=1, column=0, sticky=tk.N+tk.S+tk.E+tk.W)
        self.c_save_csv = tk.Checkbutton(self.tl_save_configs, variable=self.controller.options_menu.bool_save_csv).grid(row=1, column=1, sticky=tk.N+tk.S+tk.E+tk.W)

        self.l_save_csv_path = tk.Label(self.tl_save_configs, text='CSV path:').grid(row=1, column=2, sticky=tk.N+tk.S+tk.E+tk.W)
        self.e_save_csv_path = tk.Entry(self.tl_save_configs, textvariable=self.controller.options_menu.s_save_csv_path).grid(row=1, column=3, sticky=tk.N+tk.S+tk.E+tk.W)

        self.l_save_checkpoints = tk.Label(self.tl_save_configs, text='Save checkpoints:').grid(row=2, column=0, sticky=tk.N+tk.S+tk.E+tk.W)
        self.c_save_checkpoints = tk.Checkbutton(self.tl_save_configs, variable=self.controller.options_menu.bool_save_checkpoints).grid(row=2, column=1, sticky=tk.N+tk.S+tk.E+tk.W)

        self.l_save_checkpoints_path = tk.Label(self.tl_save_configs, text='Checkpoint path:').grid(row=2, column=2, sticky=tk.N+tk.S+tk.E+tk.W)
        self.e_save_checkpoints_path = tk.Entry(self.tl_save_configs, textvariable=self.controller.options_menu.s_save_checkpoints_path).grid(row=2, column=3, sticky=tk.N+tk.S+tk.E+tk.W)

        self.l_save_checkpoints_frequency = tk.Label(self.tl_save_configs, text='Checkpoint frequency:').grid(row=2, column=4, sticky=tk.N+tk.S+tk.E+tk.W)
        self.e_save_checkpoints_frequency = tk.Entry(self.tl_save_configs, textvariable=self.controller.options_menu.s_save_checkpoints_frequency).grid(row=2, column=5, sticky=tk.N+tk.S+tk.E+tk.W)

        self.b_tensorboard = tk.Label(self.tl_save_configs, text='Save tensorboard logs:').grid(row=3, column=0, sticky=tk.N+tk.S+tk.E+tk.W)
        self.c_tensorboard = tk.Checkbutton(self.tl_save_configs, variable=self.controller.options_menu.bool_tensorboard).grid(row=3, column=1, sticky=tk.N+tk.S+tk.E+tk.W)

        self.l_tensorboard_path = tk.Label(self.tl_save_configs, text='Tensorboard path:').grid(row=3, column=2, sticky=tk.N+tk.S+tk.E+tk.W)
        self.e_tensorboard_path = tk.Entry(self.tl_save_configs, textvariable=self.controller.options_menu.s_tensorboard_path).grid(row=3, column=3, sticky=tk.N+tk.S+tk.E+tk.W)

        self.l_tensorboard_frequency = tk.Label(self.tl_save_configs, text='Tensorboard frequency:').grid(row=3, column=4, sticky=tk.N+tk.S+tk.E+tk.W)
        self.e_tensorboard_frequency = tk.Entry(self.tl_save_configs, textvariable=self.controller.options_menu.s_tensorboard_frequency).grid(row=3, column=5, sticky=tk.N+tk.S+tk.E+tk.W)
        self.tl_save_configs.withdraw()

    def show(self):
        self.tl_save_configs.deiconify()
