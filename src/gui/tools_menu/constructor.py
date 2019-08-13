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
"""src/gui/tools_menu/constructor.py"""


import tkinter as tk
import threading
import webbrowser
import socket
import os


class ToolsMenuConstructor(tk.Menu):
    def __init__(self, parent, controller):
        tk.Menu.__init__(self, parent)
        self.parent = parent
        self.controller = controller
        self.tools_menu = tk.Menu(self.parent)
        self.parent.add_cascade(label='Tools', menu=self.tools_menu)
        self.tools_menu.add_command(label='Delete model', command=self.delete_model)
        self.tools_menu.add_separator()
        self.tools_menu.add_command(label='Delete generator', command=self.delete_gen)
        self.tools_menu.add_separator()
        self.tools_menu.add_command(label='Delete discriminator', command=self.delete_discrim)
        self.tools_menu.add_separator()
        self.tools_menu.add_command(label='Open tensorboard', command=self.open_tensorboard)

    def delete_model(self):
        self.controller.layers_list_box.delete(0, tk.END)
        self.controller.layers_list_box_serial.delete(0, tk.END)
        self.controller.home_menu.s_model_built.set('Serial model deleted')

    def delete_gen(self):
        self.controller.layers_list_box_gen.delete(0, tk.END)
        self.controller.home_menu.s_model_built.set('Generator deleted')

    def delete_discrim(self):
        self.controller.layers_list_box_discrim.delete(0, tk.END)
        self.controller.home_menu.s_model_built.set('Discriminator deleted')

    def tensorboard_clicked(self):
        tensorbaord_dir = self.controller.options_menu.s_tensorboard_path.get()
        command = 'tensorboard --logdir==' + tensorbaord_dir
        os.system(command=command)

    def open_tensorboard(self):
        local_host = socket.gethostname()
        url = 'http://' + local_host + ':6006'
        threading.Thread(target=self.tensorboard_clicked).start()
        webbrowser.open(url, new=1)

    def get_configs(self):
        pass
