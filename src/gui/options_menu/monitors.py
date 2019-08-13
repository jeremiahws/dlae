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
"""src/gui/options_menu/monitors.py"""


import tkinter as tk


class Monitors:
    def __init__(self, controller):
        self.controller = controller
        self.button_heights = 1
        self.button_widths = 15
        self.label_heights = 1
        self.label_widths = 15
        self.entry_widths = 15

        self.tl_monitors = tk.Toplevel()
        self.tl_monitors.title('Training monitors')
        self.tl_monitors.resizable(width=False, height=False)
        self.tl_monitors.wm_protocol('WM_DELETE_WINDOW', self.tl_monitors.withdraw)
        self.l_mse_monitor = tk.Label(self.tl_monitors, text='Mean squared error:').grid(row=0, column=0, sticky=tk.N+tk.S+tk.E+tk.W)
        self.c_mse_monitor = tk.Checkbutton(self.tl_monitors, variable=self.controller.options_menu.bool_mse_monitor).grid(row=0, column=1, sticky=tk.N+tk.S+tk.E+tk.W)

        self.l_mae_monitor = tk.Label(self.tl_monitors, text='Mean absolute error:').grid(row=1, column=0, sticky=tk.N+tk.S+tk.E+tk.W)
        self.c_mae_monitor = tk.Checkbutton(self.tl_monitors, variable=self.controller.options_menu.bool_mae_monitor).grid(row=1, column=1, sticky=tk.N+tk.S+tk.E+tk.W)

        self.l_acc_monitor = tk.Label(self.tl_monitors, text='Accuracy:').grid(row=2, column=0, sticky=tk.N+tk.S+tk.E+tk.W)
        self.c_acc_monitor = tk.Checkbutton(self.tl_monitors, variable=self.controller.options_menu.bool_acc_monitor).grid(row=2, column=1, sticky=tk.N+tk.S+tk.E+tk.W)
        self.tl_monitors.withdraw()

    def show(self):
        self.tl_monitors.deiconify()
