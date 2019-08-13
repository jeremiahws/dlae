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
"""src/gui/options_menu/loss_function.py"""


import tkinter as tk


class LossFunction:
    def __init__(self, controller):
        self.controller = controller
        self.button_heights = 1
        self.button_widths = 15
        self.label_heights = 1
        self.label_widths = 15
        self.entry_widths = 15

        self.tl_loss = tk.Toplevel()
        self.tl_loss.title('Loss function')
        self.tl_loss.resizable(width=False, height=False)
        self.tl_loss.wm_protocol('WM_DELETE_WINDOW', self.tl_loss.withdraw)
        self.l_loss = tk.Label(self.tl_loss, text='Loss:').grid(row=0, column=0, sticky=tk.N+tk.S+tk.E+tk.W)
        self.om_loss = tk.OptionMenu(self.tl_loss, self.controller.options_menu.s_loss, *self.controller.options_menu.o_loss)
        self.om_loss.config()
        self.om_loss.grid(row=0, column=1, sticky=tk.N+tk.S+tk.E+tk.W)

        self.l_loss_param1 = tk.Label(self.tl_loss, text='Parameter 1:').grid(row=1, column=0, sticky=tk.N+tk.S+tk.E+tk.W)
        self.e_loss_param1 = tk.Entry(self.tl_loss, textvariable=self.controller.options_menu.s_loss_param1).grid(row=1, column=1, sticky=tk.N+tk.S+tk.E+tk.W)

        self.l_loss_param2 = tk.Label(self.tl_loss, text='Parameter 2:').grid(row=2, column=0, sticky=tk.N+tk.S+tk.E+tk.W)
        self.e_loss_param2 = tk.Entry(self.tl_loss, textvariable=self.controller.options_menu.s_loss_param2).grid(row=2, column=1, sticky=tk.N+tk.S+tk.E+tk.W)
        self.tl_loss.withdraw()

    def show(self):
        self.tl_loss.deiconify()
