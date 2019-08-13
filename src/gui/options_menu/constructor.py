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
"""src/gui/options_menu/constructor.py"""


import tkinter as tk
from src.gui.options_menu.learning_rate import LearningRateSchedule
from src.gui.options_menu.loss_function import LossFunction
from src.gui.options_menu.optimizer import Optimizer
from src.gui.options_menu.train_configs import TrainingConfigurations
from src.gui.options_menu.monitors import Monitors
from src.gui.options_menu.save_configs import SaveConfigurations
from src.gui.options_menu.bbd_options import BbdOptions


class OptionsMenuConstructor(tk.Menu):
    def __init__(self, parent, controller):
        tk.Menu.__init__(self, parent)
        self.parent = parent
        self.controller = controller
        self.options_menu = tk.Menu(self.parent)
        self.parent.add_cascade(label='Options', menu=self.options_menu)

        self.learning_rate_schedule = LearningRateSchedule(self.controller)
        self.options_menu.add_command(label='Learning rate schedule', command=self.learning_rate_schedule.show)
        self.options_menu.add_separator()

        self.loss_config = LossFunction(self.controller)
        self.options_menu.add_command(label='Loss function', command=self.loss_config.show)
        self.options_menu.add_separator()

        self.optimizer = Optimizer(self.controller)
        self.options_menu.add_command(label='Optimizer', command=self.optimizer.show)
        self.options_menu.add_separator()

        self.training_configs = TrainingConfigurations(self.controller)
        self.options_menu.add_command(label='Training configurations', command=self.training_configs.show)
        self.options_menu.add_separator()

        self.monitors = Monitors(self.controller)
        self.options_menu.add_command(label='Monitors', command=self.monitors.show)
        self.options_menu.add_separator()

        self.save_configs = SaveConfigurations(self.controller)
        self.options_menu.add_command(label='Save configurations', command=self.save_configs.show)
        self.options_menu.add_separator()

        self.bbd_options = BbdOptions(self.controller)
        self.options_menu.add_command(label='BBD options', command=self.bbd_options.show)
