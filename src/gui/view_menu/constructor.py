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
"""dlae/gui/view_menu/constructor.py"""


import tkinter as tk
from src.gui.view_menu.internal_parameters import InternalParameters


class ViewMenuConstructor(tk.Menu):
    def __init__(self, parent, controller):
        tk.Menu.__init__(self, parent)
        self.parent = parent
        self.controller = controller
        self.view_menu = tk.Menu(self.parent)
        self.parent.add_cascade(label='View', menu=self.view_menu)

        self.parameters = InternalParameters(self.controller)
        self.view_menu.add_command(label='Parameter states', command=self.parameters.show)
