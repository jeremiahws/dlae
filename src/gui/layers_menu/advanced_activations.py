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
"""src/gui/layers_menu/advanced_activations.py"""


import tkinter as tk


class AdvancedActivations:
    def __init__(self, controller):
        self.controller = controller
        self.button_heights = 1
        self.button_widths = 15
        self.label_heights = 1
        self.label_widths = 15
        self.entry_widths = 15

        self.tl_adv_acts = tk.Toplevel()
        self.tl_adv_acts.title('Advanced activations')
        self.tl_adv_acts.resizable(width=False, height=False)
        self.tl_adv_acts.wm_protocol('WM_DELETE_WINDOW', self.tl_adv_acts.withdraw)
        self.b_leaky_relu = tk.Button(self.tl_adv_acts, text='Leaky reLU', command=self.b_leaky_relu_click).grid(row=0, column=0, sticky=tk.N+tk.S+tk.E+tk.W)
        self.b_thresh_relu = tk.Button(self.tl_adv_acts, text='Thresholded reLU', command=self.b_thresh_relu_click).grid(row=1, column=0, sticky=tk.N+tk.S+tk.E+tk.W)
        self.b_prelu = tk.Button(self.tl_adv_acts, text='PreLU', command=self.b_prelu_click).grid(row=2, column=0, sticky=tk.N+tk.S+tk.E+tk.W)
        self.b_elu = tk.Button(self.tl_adv_acts, text='ELU', command=self.b_elu_click).grid(row=3, column=0, sticky=tk.N+tk.S+tk.E+tk.W)

        self.l_act_param = tk.Label(self.tl_adv_acts, text='Activation parameter:').grid(row=0, column=1, sticky=tk.N+tk.S+tk.E+tk.W)
        self.e_act_param = tk.Entry(self.tl_adv_acts, textvariable=self.controller.layers_menu.s_act_param).grid(row=0, column=2, sticky=tk.N+tk.S+tk.E+tk.W)
        self.tl_adv_acts.withdraw()

    def show(self):
        self.tl_adv_acts.deiconify()

    def b_leaky_relu_click(self):
        if any(self.controller.layers_list_box.get(0)) is False or self.controller.layers_list_box.get(0).split(':')[0] != 'Input':
            self.controller.errors_list_box.insert(tk.END, 'Level2Error:FirstLayerMustBeInput')
        else:
            layer = ':'.join(['Leaky reLU', self.controller.layers_menu.s_act_param.get()])
            self.controller.layers_list_box.insert(tk.END, layer)

    def b_thresh_relu_click(self):
        if any(self.controller.layers_list_box.get(0)) is False or self.controller.layers_list_box.get(0).split(':')[0] != 'Input':
            self.controller.errors_list_box.insert(tk.END, 'Level2Error:FirstLayerMustBeInput')
        else:
            layer = ':'.join(['Thresholded reLU', self.controller.layers_menu.s_act_param.get()])
            self.controller.layers_list_box.insert(tk.END, layer)

    def b_prelu_click(self):
        if any(self.controller.layers_list_box.get(0)) is False or self.controller.layers_list_box.get(0).split(':')[0] != 'Input':
            self.controller.errors_list_box.insert(tk.END, 'Level2Error:FirstLayerMustBeInput')
        else:
            layer = 'PreLU'
            self.controller.layers_list_box.insert(tk.END, layer)

    def b_elu_click(self):
        if any(self.controller.layers_list_box.get(0)) is False or self.controller.layers_list_box.get(0).split(':')[0] != 'Input':
            self.controller.errors_list_box.insert(tk.END, 'Level2Error:FirstLayerMustBeInput')
        else:
            layer = ':'.join(['ELU', self.controller.layers_menu.s_act_param.get()])
            self.controller.layers_list_box.insert(tk.END, layer)
