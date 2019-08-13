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
"""src/gui/options_menu/learning_rate.py"""


import tkinter as tk


class LearningRateSchedule:
    def __init__(self, controller):
        self.controller = controller
        self.button_heights = 1
        self.button_widths = 15
        self.label_heights = 1
        self.label_widths = 15
        self.entry_widths = 15

        self.tl_learning_rate_schedule = tk.Toplevel()
        self.tl_learning_rate_schedule.title('Learning rate schedule')
        self.tl_learning_rate_schedule.resizable(width=False, height=False)
        self.tl_learning_rate_schedule.wm_protocol('WM_DELETE_WINDOW', self.tl_learning_rate_schedule.withdraw)
        self.l_base_lr = tk.Label(self.tl_learning_rate_schedule, text='Learning rate:').grid(row=0, column=2, sticky=tk.N+tk.S+tk.E+tk.W)
        self.e_base_lr = tk.Entry(self.tl_learning_rate_schedule, textvariable=self.controller.options_menu.s_base_lr).grid(row=0, column=3, sticky=tk.N+tk.S+tk.E+tk.W)

        self.l_lr_decay = tk.Label(self.tl_learning_rate_schedule, text='Decay:').grid(row=0, column=4, sticky=tk.N+tk.S+tk.E+tk.W)
        self.e_lr_decay = tk.Entry(self.tl_learning_rate_schedule, textvariable=self.controller.options_menu.s_lr_decay).grid(row=0, column=5, sticky=tk.N+tk.S+tk.E+tk.W)

        self.l_decay_on_plateau = tk.Label(self.tl_learning_rate_schedule, text='Decay on plateau:').grid(row=1, column=0, sticky=tk.N+tk.S+tk.E+tk.W)
        self.c_decay_on_plateau = tk.Checkbutton(self.tl_learning_rate_schedule, variable=self.controller.options_menu.bool_decay_on_plateau).grid(row=1, column=1, sticky=tk.N+tk.S+tk.E+tk.W)

        self.l_decay_on_plateau_factor = tk.Label(self.tl_learning_rate_schedule, text='Factor:').grid(row=1, column=2, sticky=tk.N+tk.S+tk.E+tk.W)
        self.e_decay_on_plateau_factor = tk.Entry(self.tl_learning_rate_schedule, textvariable=self.controller.options_menu.s_decay_on_plateau_factor).grid(row=1, column=3, sticky=tk.N+tk.S+tk.E+tk.W)

        self.l_decay_on_plateau_patience = tk.Label(self.tl_learning_rate_schedule, text='Patience:').grid(row=1, column=4, sticky=tk.N+tk.S+tk.E+tk.W)
        self.e_decay_on_plateau_patience = tk.Entry(self.tl_learning_rate_schedule, textvariable=self.controller.options_menu.s_decay_on_plateau_patience).grid(row=1, column=5, sticky=tk.N+tk.S+tk.E+tk.W)

        self.l_step_decay = tk.Label(self.tl_learning_rate_schedule, text='Step decay:').grid(row=2, column=0, sticky=tk.N+tk.S+tk.E+tk.W)
        self.c_step_decay = tk.Checkbutton(self.tl_learning_rate_schedule, variable=self.controller.options_menu.bool_step_decay).grid(row=2, column=1, sticky=tk.N+tk.S+tk.E+tk.W)

        self.l_step_decay_factor = tk.Label(self.tl_learning_rate_schedule, text='Factor:').grid(row=2, column=2, sticky=tk.N+tk.S+tk.E+tk.W)
        self.e_step_decay_factor = tk.Entry(self.tl_learning_rate_schedule, textvariable=self.controller.options_menu.s_step_decay_factor).grid(row=2, column=3, sticky=tk.N+tk.S+tk.E+tk.W)

        self.l_step_decay_period = tk.Label(self.tl_learning_rate_schedule, text='Period:').grid(row=2, column=4, sticky=tk.N+tk.S+tk.E+tk.W)
        self.e_step_decay_period = tk.Entry(self.tl_learning_rate_schedule, textvariable=self.controller.options_menu.s_step_decay_period).grid(row=2, column=5, sticky=tk.N+tk.S+tk.E+tk.W)

        self.b_d_lr = tk.Button(self.tl_learning_rate_schedule, text='Set Discriminator Learning Rate', command=self.b_d_lr_click).grid(row=3, column=0, columnspan=3, sticky=tk.N+tk.S+tk.E+tk.W)
        self.b_gan_lr = tk.Button(self.tl_learning_rate_schedule, text='Set GAN Learning Rate', command=self.b_gan_lr_click).grid(row=3, column=3, columnspan=3, sticky=tk.N+tk.S+tk.E+tk.W)
        self.tl_learning_rate_schedule.withdraw()

    def show(self):
        self.tl_learning_rate_schedule.deiconify()

    def b_d_lr_click(self):
        d_lr = ':'.join([self.controller.options_menu.s_base_lr.get(), self.controller.options_menu.s_lr_decay.get()])
        self.controller.options_menu.s_d_lr.set(d_lr)

    def b_gan_lr_click(self):
        gan_lr = ':'.join([self.controller.options_menu.s_base_lr.get(), self.controller.options_menu.s_lr_decay.get()])
        self.controller.options_menu.s_gan_lr.set(gan_lr)
