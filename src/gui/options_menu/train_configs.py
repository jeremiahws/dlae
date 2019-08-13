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
"""src/gui/options_menu/train_configs.py"""


import tkinter as tk


class TrainingConfigurations:
    def __init__(self, controller):
        self.controller = controller
        self.button_heights = 1
        self.button_widths = 15
        self.label_heights = 1
        self.label_widths = 15
        self.entry_widths = 15

        self.tl_training_configs = tk.Toplevel()
        self.tl_training_configs.title('Training configurations')
        self.tl_training_configs.resizable(width=False, height=False)
        self.tl_training_configs.wm_protocol('WM_DELETE_WINDOW', self.tl_training_configs.withdraw)
        self.l_hardware = tk.Label(self.tl_training_configs, text='Hardware:').grid(row=0, column=0, sticky=tk.N+tk.S+tk.E+tk.W)
        self.om_hardware = tk.OptionMenu(self.tl_training_configs, self.controller.options_menu.s_hardware, *self.controller.options_menu.o_hardware)
        self.om_hardware.config()
        self.om_hardware.grid(row=0, column=1, sticky=tk.N+tk.S+tk.E+tk.W)

        self.l_n_gpus = tk.Label(self.tl_training_configs, text='# GPUs:').grid(row=0, column=2, sticky=tk.N+tk.S+tk.E+tk.W)
        self.e_n_gpus = tk.Entry(self.tl_training_configs, textvariable=self.controller.options_menu.s_n_gpus).grid(row=0, column=3, sticky=tk.N+tk.S+tk.E+tk.W)

        self.l_early_stop = tk.Label(self.tl_training_configs, text='Early stop:').grid(row=1, column=0, sticky=tk.N+tk.S+tk.E+tk.W)
        self.c_early_stop = tk.Checkbutton(self.tl_training_configs, variable=self.controller.options_menu.bool_early_stop).grid(row=1, column=1, sticky=tk.N+tk.S+tk.E+tk.W)

        self.l_early_stop_patience = tk.Label(self.tl_training_configs, text='Patience:').grid(row=1, column=2, sticky=tk.N+tk.S+tk.E+tk.W)
        self.e_early_stop_patience = tk.Entry(self.tl_training_configs, textvariable=self.controller.options_menu.s_early_stop_patience).grid(row=1, column=3, sticky=tk.N+tk.S+tk.E+tk.W)

        self.l_batch_size = tk.Label(self.tl_training_configs, text='Batch size:').grid(row=2, column=2, sticky=tk.N+tk.S+tk.E+tk.W)
        self.e_batch_size = tk.Entry(self.tl_training_configs, textvariable=self.controller.options_menu.s_batch_size).grid(row=2, column=3, sticky=tk.N+tk.S+tk.E+tk.W)

        self.l_epochs = tk.Label(self.tl_training_configs, text='Epochs:').grid(row=3, column=2, sticky=tk.N+tk.S+tk.E+tk.W)
        self.e_epochs = tk.Entry(self.tl_training_configs, textvariable=self.controller.options_menu.s_epochs).grid(row=3, column=3, sticky=tk.N+tk.S+tk.E+tk.W)

        self.l_shuffle = tk.Label(self.tl_training_configs, text='Shuffle data:').grid(row=4, column=2, sticky=tk.N+tk.S+tk.E+tk.W)
        self.c_shuffle = tk.Checkbutton(self.tl_training_configs, variable=self.controller.options_menu.bool_shuffle).grid(row=4, column=3, sticky=tk.N+tk.S+tk.E+tk.W)

        self.l_val_split = tk.Label(self.tl_training_configs, text='Validation split:').grid(row=5, column=2, sticky=tk.N+tk.S+tk.E+tk.W)
        self.e_val_split = tk.Entry(self.tl_training_configs, textvariable=self.controller.options_menu.s_val_split).grid(row=5, column=3, sticky=tk.N+tk.S+tk.E+tk.W)
        self.tl_training_configs.withdraw()

    def show(self):
        self.tl_training_configs.deiconify()
