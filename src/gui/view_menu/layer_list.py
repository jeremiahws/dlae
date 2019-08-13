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
"""dlae/gui/view_menu/layer_list.py"""


import tkinter as tk


class LayerList:
    def __init__(self):
        self.button_heights = 1
        self.button_widths = 15
        self.label_heights = 1
        self.label_widths = 15
        self.entry_widths = 15

        self.tl_layer_list = tk.Toplevel()
        self.tl_layer_list.title('List of layers')
        self.tl_layer_list.wm_protocol('WM_DELETE_WINDOW', self.tl_layer_list.withdraw)
        self.tl_layer_list.resizable(width=False, height=False)

        self.b_serial_layers = tk.Button(self.tl_layer_list, text='View serial layers', command=self.view_serial_layers, height=self.button_heights).grid(row=0, column=0, columnspan=3, sticky=tk.E+tk.W)
        self.b_gen_layers = tk.Button(self.tl_layer_list, text='View generator layers', command=self.view_gen_layers, height=self.button_heights).grid(row=0, column=3, columnspan=3, sticky=tk.E+tk.W)
        self.b_discrim_layers = tk.Button(self.tl_layer_list, text='View discriminator layers', command=self.view_discrim_layers, height=self.button_heights).grid(row=0, column=6, columnspan=3, sticky=tk.E+tk.W)
        self.b_serial_layers = tk.Button(self.tl_layer_list, text='Rebuild model', command=self.rebuild_serial_layers, height=self.button_heights).grid(row=2, column=0, columnspan=3, sticky=tk.E+tk.W)
        self.b_gen_layers = tk.Button(self.tl_layer_list, text='Rebuild generator', command=self.rebuild_gen_layers, height=self.button_heights).grid(row=2, column=3, columnspan=3, sticky=tk.E+tk.W)
        self.b_discrim_layers = tk.Button(self.tl_layer_list, text='Rebuild discriminator', command=self.rebuild_discrim_layers, height=self.button_heights).grid(row=2, column=6, columnspan=3, sticky=tk.E+tk.W)

        self.lb_layers_list = tk.Listbox(self.tl_layer_list)
        self.lb_layers_list.bind('<<ListboxSelect>>', self.cursor_select)
        self.lb_layers_list.config(width=85, height=25)
        self.lb_layers_list.grid(row=1, column=0, columnspan=9, sticky=tk.N+tk.S+tk.E+tk.W)
        self.sb_layers_list = tk.Scrollbar(self.tl_layer_list, orient="vertical")
        self.sb_layers_list.config(command=self.lb_layers_list.yview)
        self.sb_layers_list.grid(row=1, column=9, sticky=tk.N+tk.S)
        self.lb_layers_list.config(yscrollcommand=self.sb_layers_list.set)

        self.lb_layers_list_serial = tk.Listbox(self.tl_layer_list)
        self.lb_layers_list_gen = tk.Listbox(self.tl_layer_list)
        self.lb_layers_list_discrim = tk.Listbox(self.tl_layer_list)

        self.s_layer_to_modify = tk.StringVar(value="No layer selected")
        self.i_index = tk.IntVar()
        self.b_layer_to_modify = tk.Button(self.tl_layer_list, text='Update layer', command=self.change_layer, height=self.button_heights).grid(row=3, column=0, columnspan=3, sticky=tk.E+tk.W)
        self.b_inject_layer = tk.Button(self.tl_layer_list, text='Inject layer', command=self.inject_layer, height=self.button_heights).grid(row=3, column=3, columnspan=3, sticky=tk.E+tk.W)
        self.b_delete_layer = tk.Button(self.tl_layer_list, text='Delete layer', command=self.delete_layer, height=self.button_heights).grid(row=3, column=6, columnspan=3, sticky=tk.E+tk.W)
        self.e_layer_to_modify = tk.Entry(self.tl_layer_list, textvariable=self.s_layer_to_modify, width=self.entry_widths).grid(row=4, column=0, columnspan=9, sticky=tk.E+tk.W)

        self.tl_layer_list.withdraw()

    def cursor_select(self, event):
        try:
            index = self.lb_layers_list.curselection()[0]
            selection = self.lb_layers_list.get(index)
            self.i_index.set(index)
            self.s_layer_to_modify.set(selection)
        except:
            pass

    def change_layer(self):
        self.lb_layers_list.delete(self.i_index.get())
        self.lb_layers_list.insert(self.i_index.get(), self.s_layer_to_modify.get())

    def inject_layer(self):
        self.lb_layers_list.insert(self.i_index.get() + 1, self.s_layer_to_modify.get())

    def delete_layer(self):
        self.lb_layers_list.delete(self.i_index.get())

    def show(self):
        self.tl_layer_list.deiconify()

    def view_serial_layers(self):
        layers = self.lb_layers_list_serial.get(0, tk.END)

        if any(layers):
            self.lb_layers_list.delete(0, tk.END)
            [self.lb_layers_list.insert(tk.END, layer) for layer in layers]
        else:
            self.lb_layers_list.delete(0, tk.END)

    def view_gen_layers(self):
        layers = self.lb_layers_list_gen.get(0, tk.END)

        if any(layers):
            self.lb_layers_list.delete(0, tk.END)
            [self.lb_layers_list.insert(tk.END, layer) for layer in layers]
        else:
            self.lb_layers_list.delete(0, tk.END)

    def view_discrim_layers(self):
        layers = self.lb_layers_list_discrim.get(0, tk.END)

        if any(layers):
            self.lb_layers_list.delete(0, tk.END)
            [self.lb_layers_list.insert(tk.END, layer) for layer in layers]
        else:
            self.lb_layers_list.delete(0, tk.END)

    def rebuild_serial_layers(self):
        layers = self.lb_layers_list.get(0, tk.END)
        self.lb_layers_list_serial.delete(0, tk.END)
        [self.lb_layers_list_serial.insert(tk.END, layer) for layer in layers]
        self.lb_layers_list.delete(0, tk.END)

    def rebuild_gen_layers(self):
        layers = self.lb_layers_list.get(0, tk.END)
        self.lb_layers_list_gen.delete(0, tk.END)
        [self.lb_layers_list_gen.insert(tk.END, layer) for layer in layers]
        self.lb_layers_list.delete(0, tk.END)

    def rebuild_discrim_layers(self):
        layers = self.lb_layers_list.get(0, tk.END)
        self.lb_layers_list_discrim.delete(0, tk.END)
        [self.lb_layers_list_discrim.insert(tk.END, layer) for layer in layers]
        self.lb_layers_list.delete(0, tk.END)
