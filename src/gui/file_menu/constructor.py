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
"""src/gui/file_menu/constructor.py"""


import tkinter as tk
import json
from tkinter import filedialog
import uuid
import tensorflow as tf


class FileMenuConstructor(tk.Menu):
    def __init__(self, parent, controller):
        tk.Menu.__init__(self, parent)
        self.parent = parent
        self.controller = controller
        self.file_menu = tk.Menu(self.parent)
        self.parent.add_cascade(label='File', menu=self.file_menu)
        self.file_menu.add_command(label='Load config file', command=self.load_config)
        self.file_menu.add_separator()
        self.file_menu.add_command(label='Save config file', command=self.save_config)
        self.file_menu.add_separator()
        self.file_menu.add_command(label='Load model checkpoint', command=self.load_checkpoint)
        self.file_menu.add_separator()
        self.file_menu.add_command(label='Load model', command=self.load_model)
        self.file_menu.add_separator()

        self.file_menu.add_command(label='Exit', command=self.parent.quit)

    def load_config(self):
        tf.reset_default_graph()
        self.controller.file_menu.load_file_path = tk.filedialog.askopenfile(filetypes=(('JSON files', '*.json'), ('All files', '*.*')))
        if self.controller.file_menu.load_file_path is None:
            return
        else:
            self.controller.file_menu.s_load_file_path.set(self.controller.file_menu.load_file_path.name)
            with open(self.controller.file_menu.s_load_file_path.get(), 'r') as f:
                configs = json.load(f)

            f.close()

            self.controller.set_configs(configs)

    def save_config(self):
        self.controller.file_menu.save_file_path = tk.filedialog.asksaveasfilename(filetypes=(('JSON files', '*.json'), ('All files', '*.*')))
        if self.controller.file_menu.save_file_path is None or any(self.controller.file_menu.save_file_path) is False:
            return
        else:
            configs = self.controller.get_configs()

            if len(self.controller.file_menu.save_file_path.split('.json')) == 2:
                pass
            else:
                self.controller.file_menu.save_file_path += '.json'

            with open(self.controller.file_menu.save_file_path, 'w') as f:
                json.dump(configs, f, indent=1)

            f.close()

    def save_config_hotkey(self, event):
        if any(self.controller.file_menu.s_load_file_path.get()):
            self.controller.file_menu.save_file_path = self.controller.file_menu.s_load_file_path.get()
            configs = self.controller.get_configs()

            with open(self.controller.file_menu.save_file_path, 'w') as f:
                json.dump(configs, f, indent=1)

        else:
            self.controller.file_menu.save_file_path = uuid.uuid4().hex
            configs = self.controller.get_configs()
            self.controller.file_menu.save_file_path += '.json'

            with open(self.controller.file_menu.save_file_path, 'w') as f:
                json.dump(configs, f, indent=1)

        f.close()

    def load_checkpoint(self):
        self.controller.file_menu.load_ckpt_file_path = tk.filedialog.askopenfile(filetypes=(('HDF5 files', '*.h5'), ('All files', '*.*')))
        if self.controller.file_menu.load_ckpt_file_path is None:
            return
        else:
            self.controller.file_menu.s_load_ckpt_file_path.set(self.controller.file_menu.load_ckpt_file_path.name)

    def load_model(self):
        self.controller.file_menu.load_model_file_path = tk.filedialog.askopenfile(filetypes=(('HDF5 files', '*.h5'), ('All files', '*.*')))
        if self.controller.file_menu.load_model_file_path is None:
            return
        else:
            self.controller.file_menu.s_load_model_file_path.set(self.controller.file_menu.load_model_file_path.name)
