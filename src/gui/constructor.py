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
"""src/gui/constructor.py"""


import tkinter as tk
from src.gui.file_menu.constructor import FileMenuConstructor
from src.gui.data_menu.constructor import DataMenuConstructor
from src.gui.view_menu.constructor import ViewMenuConstructor
from src.gui.layers_menu.constructor import LayersMenuConstructor
from src.gui.options_menu.constructor import OptionsMenuConstructor
from src.gui.tools_menu.constructor import ToolsMenuConstructor
from src.gui.run_menu.constructor import RunMenuConstructor
from src.gui.help_menu.constructor import HelpMenuConstructor
from src.gui.home_menu.constructor import HomeMenuConstructor
from src.utils.gui_utils import GuiParameterController


class DlaeGui(tk.Frame):
    def __init__(self, parent):
        """
        Constructor class to build the graphical user interface for DLAE.
        :param parent: tkinter root

        Attributes:
            parent -- tkinter root
            menu_seed -- lowest level parent menu
            controller -- constructs and handles all editable GUI variables
            file_menu -- constructs GUI File menu on top of menu_seed
            data_menu -- constructs GUI Data menu on top of menu_seed
            view_menu -- constructs GUI View menu on top of menu_seed
            layers_menu -- constructs GUI Layers menu on top of menu_seed
            options_menu -- constructs GUI Options menu on top of menu_seed
            tools_menu -- constructs GUI Tools menu on top of menu_seed
            run_menu -- constructs GUI Run menu on top of menu_seed
            help_menu -- constructs GUI Help menu on top of menu_seed
            home_menu -- constructs GUI Home menu (it's part of the menu_seed)
        """
        tk.Frame.__init__(self, parent)
        self.parent = parent
        self.parent.title('Deep Learning Application Engine (DLAE)')
        self.menu_seed = tk.Menu(self.parent)

        self.controller = GuiParameterController()
        self.home_menu = HomeMenuConstructor(self.parent, self.controller)
        self.controller.layers_list_box = self.home_menu.lb_layers_list
        self.controller.layers_list_box_serial = self.home_menu.lb_layers_list_serial
        self.controller.layers_list_box_gen = self.home_menu.lb_layers_list_gen
        self.controller.layers_list_box_discrim = self.home_menu.lb_layers_list_discrim
        self.file_menu = FileMenuConstructor(self.menu_seed, self.controller)
        self.data_menu = DataMenuConstructor(self.menu_seed, self.controller)
        self.view_menu = ViewMenuConstructor(self.menu_seed, self.controller)
        self.layers_menu = LayersMenuConstructor(self.menu_seed, self.controller)
        self.options_menu = OptionsMenuConstructor(self.menu_seed, self.controller)
        self.tools_menu = ToolsMenuConstructor(self.menu_seed, self.controller)
        self.run_menu = RunMenuConstructor(self.menu_seed, self.controller)
        self.help_menu = HelpMenuConstructor(self.menu_seed)
        self.controller.errors_list_box = self.help_menu.errors_list.lb_errors

        # Set hotkeys
        self.bind_all("<Control-s>", self.file_menu.save_config_hotkey)

        # Required to set menu bar
        self.parent.config(menu=self.menu_seed)
