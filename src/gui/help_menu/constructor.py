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
"""src/gui/help_menu/constructor.py"""


import tkinter as tk
import webbrowser
from src.gui.help_menu.errors import Errors


class HelpMenuConstructor(tk.Menu):
    def __init__(self, parent):
        tk.Menu.__init__(self, parent)
        self.parent = parent
        self.help_menu = tk.Menu(self.parent)
        self.parent.add_cascade(label='Help', menu=self.help_menu)

        self.help_menu.add_command(label='DLAE Github', command=self.github_clicked)
        self.help_menu.add_separator()

        self.errors_list = Errors()
        self.help_menu.add_command(label='Error log', command=self.errors_list.show)

    @staticmethod
    def github_clicked():
        url = 'http://dlae.io'
        webbrowser.open(url, new=1)
