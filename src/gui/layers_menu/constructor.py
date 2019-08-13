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
"""src/gui/layers_menu/constructor.py"""


import tkinter as tk
from src.gui.layers_menu.convolutional_layers import ConvolutionalLayers
from src.gui.layers_menu.pooling_layers import PoolingLayers
from src.gui.layers_menu.utility_layers import UtilityLayers
from src.gui.layers_menu.advanced_activations import AdvancedActivations
from src.gui.layers_menu.pretrained_networks import PretrainedNetworks


class LayersMenuConstructor(tk.Menu):
    def __init__(self, parent, controller):
        tk.Menu.__init__(self, parent)
        self.parent = parent
        self.controller = controller
        self.layers_menu = tk.Menu(self.parent)
        self.parent.add_cascade(label='Layers', menu=self.layers_menu)

        self.conv_layers = ConvolutionalLayers(self.controller)
        self.layers_menu.add_command(label='Convolution layers', command=self.conv_layers.show)
        self.layers_menu.add_separator()

        self.pool_layers = PoolingLayers(self.controller)
        self.layers_menu.add_command(label='Pooling layers', command=self.pool_layers.show)
        self.layers_menu.add_separator()

        self.util_layers = UtilityLayers(self.controller)
        self.layers_menu.add_command(label='Utility layers', command=self.util_layers.show)
        self.layers_menu.add_separator()

        self.adv_acts = AdvancedActivations(self.controller)
        self.layers_menu.add_command(label='Advanced activations', command=self.adv_acts.show)
        self.layers_menu.add_separator()

        self.pretrained_nets = PretrainedNetworks(self.controller)
        self.layers_menu.add_command(label='Pretrained networks', command=self.pretrained_nets.show)
