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
"""src/gui/options_menu/optimizer.py"""


import tkinter as tk


class Optimizer:
    def __init__(self, controller):
        self.controller = controller
        self.button_heights = 1
        self.button_widths = 15
        self.label_heights = 1
        self.label_widths = 15
        self.entry_widths = 15

        self.tl_optimizer = tk.Toplevel()
        self.tl_optimizer.title('Optimizer configurations')
        self.tl_optimizer.resizable(width=False, height=False)
        self.tl_optimizer.wm_protocol('WM_DELETE_WINDOW', self.tl_optimizer.withdraw)
        self.l_optimizer = tk.Label(self.tl_optimizer, text='Optimizer:').grid(row=0, column=0, sticky=tk.N+tk.S+tk.E+tk.W)
        self.om_optimizer = tk.OptionMenu(self.tl_optimizer, self.controller.options_menu.s_optimizer, *self.controller.options_menu.o_optimizer)
        self.om_optimizer.config()
        self.om_optimizer.grid(row=0, column=1, sticky=tk.N+tk.S+tk.E+tk.W)

        self.l_optimizer_beta1 = tk.Label(self.tl_optimizer, text='Beta 1:').grid(row=0, column=2, sticky=tk.N+tk.S+tk.E+tk.W)
        self.e_optimizer_beta1 = tk.Entry(self.tl_optimizer, textvariable=self.controller.options_menu.s_optimizer_beta1).grid(row=0, column=3, sticky=tk.N+tk.S+tk.E+tk.W)

        self.l_optimizer_beta2 = tk.Label(self.tl_optimizer, text='Beta 2:').grid(row=1, column=2, sticky=tk.N+tk.S+tk.E+tk.W)
        self.e_optimizer_beta2 = tk.Entry(self.tl_optimizer, textvariable=self.controller.options_menu.s_optimizer_beta2).grid(row=1, column=3, sticky=tk.N+tk.S+tk.E+tk.W)

        self.l_optimizer_rho = tk.Label(self.tl_optimizer, text='Rho:').grid(row=2, column=2, sticky=tk.N+tk.S+tk.E+tk.W)
        self.e_optimizer_rho = tk.Entry(self.tl_optimizer, textvariable=self.controller.options_menu.s_optimizer_rho).grid(row=2, column=3, sticky=tk.N+tk.S+tk.E+tk.W)

        self.l_optimizer_momentum = tk.Label(self.tl_optimizer, text='Momentum:').grid(row=3, column=2, sticky=tk.N+tk.S+tk.E+tk.W)
        self.e_optimizer_momentum = tk.Entry(self.tl_optimizer, textvariable=self.controller.options_menu.s_optimizer_momentum).grid(row=3, column=3, sticky=tk.N+tk.S+tk.E+tk.W)

        self.l_optimizer_epsilon = tk.Label(self.tl_optimizer, text='Epsilon:').grid(row=4, column=2, sticky=tk.N+tk.S+tk.E+tk.W)
        self.e_optimizer_epsilon = tk.Entry(self.tl_optimizer, textvariable=self.controller.options_menu.s_optimizer_epsilon).grid(row=4, column=3, sticky=tk.N+tk.S+tk.E+tk.W)

        self.b_d_optimizer = tk.Button(self.tl_optimizer, text='Set Discriminator Optimizer', command=self.b_d_optimizer_click).grid(row=3, column=0, columnspan=2, sticky=tk.N+tk.S+tk.E+tk.W)
        self.b_gan_optimizer = tk.Button(self.tl_optimizer, text='Set GAN Optimizer', command=self.b_gan_optimizer_click).grid(row=4, column=0, columnspan=2, sticky=tk.N+tk.S+tk.E+tk.W)
        self.tl_optimizer.withdraw()

    def show(self):
        self.tl_optimizer.deiconify()

    def b_d_optimizer_click(self):
        d_optimizer = ':'.join([self.controller.options_menu.s_optimizer.get(), self.controller.options_menu.s_optimizer_beta1.get(), self.controller.options_menu.s_optimizer_beta2.get(), self.controller.options_menu.s_optimizer_rho.get(), self.controller.options_menu.s_optimizer_momentum.get(), self.controller.options_menu.s_optimizer_epsilon.get()])
        self.controller.options_menu.s_d_optimizer.set(d_optimizer)

    def b_gan_optimizer_click(self):
        gan_optimizer = ':'.join([self.controller.options_menu.s_optimizer.get(), self.controller.options_menu.s_optimizer_beta1.get(), self.controller.options_menu.s_optimizer_beta2.get(), self.controller.options_menu.s_optimizer_rho.get(), self.controller.options_menu.s_optimizer_momentum.get(), self.controller.options_menu.s_optimizer_epsilon.get()])
        self.controller.options_menu.s_gan_optimizer.set(gan_optimizer)
