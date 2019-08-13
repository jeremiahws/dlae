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
"""src/gui/help_menu/errors.py"""


import tkinter as tk


class Errors:
    def __init__(self):
        self.button_heights = 1
        self.button_widths = 15
        self.label_heights = 1
        self.label_widths = 15
        self.entry_widths = 15

        self.tl_errors = tk.Toplevel()
        self.tl_errors.title('Errors')
        self.tl_errors.wm_protocol('WM_DELETE_WINDOW', self.tl_errors.withdraw)

        self.lb_errors = tk.Listbox(self.tl_errors)
        self.lb_errors.config(width=85, height=25)
        self.lb_errors.grid(row=0, column=0, sticky=tk.N+tk.S+tk.E+tk.W)
        self.sb_errors = tk.Scrollbar(self.tl_errors, orient="vertical")
        self.sb_errors.config(command=self.lb_errors.yview)
        self.sb_errors.grid(row=0, column=1, sticky=tk.N+tk.S)
        self.lb_errors.config(yscrollcommand=self.sb_errors.set)

        self.tl_errors.resizable(width=False, height=False)
        self.tl_errors.withdraw()

    def show(self):
        self.tl_errors.deiconify()
