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
"""src/gui/data_menu/constructor.py"""


import tkinter as tk
from tkinter import filedialog
from src.gui.data_menu.preprocessing import Preprocessing
from src.gui.data_menu.augmentation import Augmentation


class DataMenuConstructor(tk.Menu):
    def __init__(self, parent, controller):
        """
        Constructor class to build the data menu for the graphical user interface.
        :param parent: lowest level parent menu
        :param controller: the GUI variable controller

        Attributes:
            parent -- lowest level parent menu
            controller -- constructs and handles all editable GUI variables
            data_menu -- data_menu built on the parent
            preprocessing -- pop-out menu for preprocessing steps
            postprocessing -- pop-out menu for postprocessing steps
        """
        tk.Menu.__init__(self, parent)
        self.parent = parent
        self.controller = controller
        self.data_menu = tk.Menu(self.parent)
        self.parent.add_cascade(label='Data', menu=self.data_menu)

        self.data_menu.add_command(label='Load train X', command=self.load_train_X)
        self.data_menu.add_separator()
        self.data_menu.add_command(label='Load train y', command=self.load_train_y)
        self.data_menu.add_separator()
        self.data_menu.add_command(label='Load validation X', command=self.load_val_X)
        self.data_menu.add_separator()
        self.data_menu.add_command(label='Load validation y', command=self.load_val_y)
        self.data_menu.add_separator()
        self.data_menu.add_command(label='Load test X', command=self.load_test_X)
        self.data_menu.add_separator()

        self.preprocessing = Preprocessing(self.controller)
        self.data_menu.add_command(label='Preprocessing', command=self.preprocessing.show)
        self.data_menu.add_separator()

        self.augmentation = Augmentation(self.controller)
        self.data_menu.add_command(label='Augmentation', command=self.augmentation.show)

    def load_train_X(self):
        self.controller.data_menu.train_X_path = tk.filedialog.askopenfile(filetypes=(('HDF5 files', '*.h5'), ('All files', '*.*')))
        if self.controller.data_menu.train_X_path is None:
            return

        else:
            self.controller.data_menu.train_X_path = self.controller.data_menu.train_X_path.name
            self.controller.data_menu.s_train_X_path.set(self.controller.data_menu.train_X_path)

    def load_train_y(self):
        self.controller.data_menu.train_y_path = tk.filedialog.askopenfile(filetypes=(('HDF5 files', '*.h5'), ('All files', '*.*')))
        if self.controller.data_menu.train_y_path is None:
            return

        else:
            self.controller.data_menu.train_y_path = self.controller.data_menu.train_y_path.name
            self.controller.data_menu.s_train_y_path.set(self.controller.data_menu.train_y_path)

    def load_val_X(self):
        self.controller.data_menu.val_X_path = tk.filedialog.askopenfile(filetypes=(('HDF5 files', '*.h5'), ('All files', '*.*')))
        if self.controller.data_menu.val_X_path is None:
            return

        else:
            self.controller.data_menu.val_X_path = self.controller.data_menu.val_X_path.name
            self.controller.data_menu.s_val_X_path.set(self.controller.data_menu.val_X_path)

    def load_val_y(self):
        self.controller.data_menu.val_y_path = tk.filedialog.askopenfile(filetypes=(('HDF5 files', '*.h5'), ('All files', '*.*')))
        if self.controller.data_menu.val_y_path is None:
            return

        else:
            self.controller.data_menu.val_y_path = self.controller.data_menu.val_y_path.name
            self.controller.data_menu.s_val_y_path.set(self.controller.data_menu.val_y_path)

    def load_test_X(self):
        self.controller.data_menu.test_X_path = tk.filedialog.askopenfile(filetypes=(('HDF5 files', '*.h5'), ('All files', '*.*')))
        if self.controller.data_menu.test_X_path is None:
            return

        else:
            self.controller.data_menu.test_X_path = self.controller.data_menu.test_X_path.name
            self.controller.data_menu.s_test_X_path.set(self.controller.data_menu.test_X_path)

    def load_test_y(self):
        self.controller.data_menu.test_y_path = tk.filedialog.askopenfile(filetypes=(('HDF5 files', '*.h5'), ('All files', '*.*')))
        if self.controller.data_menu.test_y_path is None:
            return

        else:
            self.controller.data_menu.test_y_path = self.controller.data_menu.test_y_path.name
            self.controller.data_menu.s_test_y_path.set(self.controller.data_menu.test_y_path)
