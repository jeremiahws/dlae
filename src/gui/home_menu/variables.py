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
"""src/gui/home_menu/variables.py"""


import tkinter as tk


class HomeMenuVariables(object):
    def __init__(self):
        self.p_prebuilt_cnn = tk.PhotoImage(file='src/gui/button_figs/home/prebuilt_cnn.png')
        self.p_prebuilt_fcn = tk.PhotoImage(file='src/gui/button_figs/home/prebuilt_fcn.png')
        self.p_prebuilt_gan = tk.PhotoImage(file='src/gui/button_figs/home/prebuilt_gan.png')
        self.p_prebuilt_bbd = tk.PhotoImage(file='src/gui/button_figs/home/prebuilt_bbd.png')

        self.o_model_signal = ('CNN', 'FCN', 'GAN', 'BBD')
        self.s_model_signal = tk.StringVar(value=self.o_model_signal[0])

        self.o_type_signal = ('Train', 'Train from Checkpoint', 'Inference')
        self.s_type_signal = tk.StringVar(value=self.o_type_signal[0])

        self.s_input_shape = tk.StringVar(value="(512, 512, 1)")

        self.s_model_compile = tk.StringVar(value="N/A")
        self.s_model_built = tk.StringVar(value="Model not built")
