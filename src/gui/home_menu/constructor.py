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
"""src/gui/home_menu/constructor.py"""


import tkinter as tk
from src.gui.home_menu.prebuilt_cnn import PrebuiltConvolutionalNeuralNetwork
from src.gui.home_menu.prebuilt_fcn import PrebuiltFullyConvolutionalNetwork
from src.gui.home_menu.prebuilt_gan import PrebuiltGenerativeAdversarialNetwork
from src.gui.home_menu.prebuilt_bbd import PrebuiltBoundingBoxDetector
from src.utils.engine_utils import get_io
from keras.models import Model
from keras.utils import plot_model
import datetime
import os
import time
import tensorflow as tf
from src.utils.ssd_utils import *
from ast import literal_eval


class HomeMenuConstructor(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)

        self.button_heights = 1
        self.button_widths = 15
        self.label_heights = 1
        self.label_widths = 15
        self.entry_widths = 15
        self.controller = controller

        self.om_model_signal = tk.OptionMenu(parent, self.controller.home_menu.s_model_signal, *self.controller.home_menu.o_model_signal)
        self.om_model_signal.config()
        self.om_model_signal.grid(row=0, column=0, sticky=tk.E+tk.W)

        self.om_type_signal = tk.OptionMenu(parent, self.controller.home_menu.s_type_signal, *self.controller.home_menu.o_type_signal)
        self.om_type_signal.config()
        self.om_type_signal.grid(row=0, column=1, sticky=tk.E+tk.W)

        self.l_input_shape = tk.Label(parent, text='Input shape:').grid(row=0, column=2, sticky=tk.E+tk.W)
        self.e_input_shape = tk.Entry(parent, textvariable=self.controller.home_menu.s_input_shape).grid(row=0, column=3, sticky=tk.N+tk.S+tk.E+tk.W)
        
        self.prebuilt_cnn = PrebuiltConvolutionalNeuralNetwork(self.controller)
        self.b_prebuilt_cnn = tk.Button(parent, image=self.controller.home_menu.p_prebuilt_cnn, command=self.prebuilt_cnn.show).grid(row=1, column=0)

        self.prebuilt_fcn = PrebuiltFullyConvolutionalNetwork(self.controller)
        self.b_prebuilt_fcn = tk.Button(parent, image=self.controller.home_menu.p_prebuilt_fcn, command=self.prebuilt_fcn.show).grid(row=1, column=1)

        self.prebuilt_gan = PrebuiltGenerativeAdversarialNetwork(self.controller)
        self.b_prebuilt_gan = tk.Button(parent, image=self.controller.home_menu.p_prebuilt_gan, command=self.prebuilt_gan.show).grid(row=1, column=2)

        self.prebuilt_bbd = PrebuiltBoundingBoxDetector(self.controller)
        self.b_prebuilt_bbd = tk.Button(parent, image=self.controller.home_menu.p_prebuilt_bbd, command=self.prebuilt_bbd.show).grid(row=1, column=3)

        self.b_serial_layers = tk.Button(parent, text='View serial layers', command=self.view_serial_layers, height=self.button_heights).grid(row=2, column=0, sticky=tk.E+tk.W)
        self.b_gen_layers = tk.Button(parent, text='View generator layers', command=self.view_gen_layers, height=self.button_heights).grid(row=2, column=1, sticky=tk.E+tk.W)
        self.b_discrim_layers = tk.Button(parent, text='View discriminator layers', command=self.view_discrim_layers, height=self.button_heights).grid(row=2, column=2, sticky=tk.E+tk.W)
        self.e_model_built = tk.Entry(parent, textvariable=self.controller.home_menu.s_model_built, width=self.entry_widths).grid(row=2, column=3, sticky=tk.N+tk.S+tk.E+tk.W)
        self.b_serial_layers = tk.Button(parent, text='Build/rebuild model', command=self.rebuild_serial_layers, height=self.button_heights).grid(row=4, column=0, sticky=tk.E+tk.W)
        self.b_gen_layers = tk.Button(parent, text='Build/rebuild generator', command=self.rebuild_gen_layers, height=self.button_heights).grid(row=4, column=1, sticky=tk.E+tk.W)
        self.b_discrim_layers = tk.Button(parent, text='Build/rebuild discriminator', command=self.rebuild_discrim_layers, height=self.button_heights).grid(row=4, column=2, sticky=tk.E+tk.W)
        self.b_test_model_compile = tk.Button(parent, text='Test model compilation', command=self.test_model_compile, height=self.button_heights).grid(row=4, column=3, sticky=tk.E+tk.W)

        self.lb_layers_list = tk.Listbox(parent)
        self.lb_layers_list.bind('<<ListboxSelect>>', self.cursor_select)
        self.lb_layers_list.config(width=85, height=25)
        self.lb_layers_list.grid(row=3, column=0, columnspan=4, sticky=tk.N+tk.S+tk.E+tk.W)
        self.sb_layers_list = tk.Scrollbar(parent, orient="vertical")
        self.sb_layers_list.config(command=self.lb_layers_list.yview)
        self.sb_layers_list.grid(row=3, column=5, sticky=tk.N+tk.S)
        self.lb_layers_list.config(yscrollcommand=self.sb_layers_list.set)

        self.lb_layers_list_serial = tk.Listbox(parent)
        self.lb_layers_list_gen = tk.Listbox(parent)
        self.lb_layers_list_discrim = tk.Listbox(parent)

        self.s_layer_to_modify = tk.StringVar(value="No layer selected")
        self.i_index = tk.IntVar()
        self.b_layer_to_modify = tk.Button(parent, text='Update layer', command=self.change_layer, height=self.button_heights).grid(row=5, column=0, sticky=tk.E+tk.W)
        self.b_inject_layer = tk.Button(parent, text='Inject layer', command=self.inject_layer, height=self.button_heights).grid(row=5, column=1, sticky=tk.E+tk.W)
        self.b_delete_layer = tk.Button(parent, text='Delete layer', command=self.delete_layer, height=self.button_heights).grid(row=5, column=2, sticky=tk.E+tk.W)
        self.e_layer_to_modify = tk.Entry(parent, textvariable=self.controller.home_menu.s_model_compile, width=self.entry_widths).grid(row=5, column=3, sticky=tk.N+tk.S+tk.E+tk.W)
        self.e_test_model_compile = tk.Entry(parent, textvariable=self.s_layer_to_modify, width=self.entry_widths).grid(row=6, column=0, columnspan=4, sticky=tk.E+tk.W)

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
        self.controller.home_menu.s_model_built.set('Serial model built')

    def rebuild_gen_layers(self):
        layers = self.lb_layers_list.get(0, tk.END)
        self.lb_layers_list_gen.delete(0, tk.END)
        [self.lb_layers_list_gen.insert(tk.END, layer) for layer in layers]
        self.lb_layers_list.delete(0, tk.END)
        self.controller.home_menu.s_model_built.set('Generator built')

    def rebuild_discrim_layers(self):
        layers = self.lb_layers_list.get(0, tk.END)
        self.lb_layers_list_discrim.delete(0, tk.END)
        [self.lb_layers_list_discrim.insert(tk.END, layer) for layer in layers]
        self.lb_layers_list.delete(0, tk.END)
        self.controller.home_menu.s_model_built.set('Discriminator built')

    def test_model_compile(self):
        tf.reset_default_graph()
        layers = self.controller.layers_list_box.get(0, tk.END)
        if any(layers):
            inputs, outputs, hooks, errors = get_io(layers)
            [self.controller.errors_list_box.insert(tk.END, error) for error in errors]
            try:
                if self.controller.home_menu.s_model_signal.get() == 'BBD':
                    input_shape = literal_eval(self.controller.home_menu.s_input_shape.get())
                    n_predictor_layers = len(hooks)
                    class_layers = []
                    box_layers = []
                    anchor_layers = []
                    class_layers_reshape = []
                    box_layers_reshape = []
                    anchor_layers_reshape = []
                    predictor_sizes = []
                    scales = np.linspace(0.1, 0.9, n_predictor_layers + 1)
                    aspect_ratios = [0.5, 1.0, 2.0]
                    steps = [None] * n_predictor_layers
                    offsets = [None] * n_predictor_layers
                    n_boxes = [3] * n_predictor_layers
                    for j, hook in enumerate(hooks):
                        if j == 0:
                            hook = L2Normalization()(hook)
                        new_c_layer = keras.layers.Conv2D(n_boxes[j] * (1 + 1), (3, 3), strides=(1, 1), padding='same',
                                                          kernel_initializer='he_normal')(hook)
                        new_b_layer = keras.layers.Conv2D(n_boxes[j] * 4, (3, 3), strides=(1, 1), padding='same',
                                                          kernel_initializer='he_normal')(hook)
                        new_a_layer = AnchorBoxes(input_shape[0],
                                                  input_shape[1],
                                                  this_scale=scales[j],
                                                  next_scale=scales[j + 1],
                                                  aspect_ratios=aspect_ratios,
                                                  two_boxes_for_ar1=False,
                                                  this_steps=steps[j],
                                                  this_offsets=offsets[j],
                                                  clip_boxes=False,
                                                  variances=[1.0, 1.0, 1.0, 1.0],
                                                  coords='centroids',
                                                  normalize_coords=True)(new_b_layer)
                        class_layers.append(new_c_layer)
                        box_layers.append(new_b_layer)
                        anchor_layers.append(new_a_layer)
                        class_layers_reshape.append(keras.layers.Reshape((-1, (1 + 1)))(class_layers[-1]))
                        box_layers_reshape.append(keras.layers.Reshape((-1, 4))(box_layers[-1]))
                        anchor_layers_reshape.append(keras.layers.Reshape((-1, 8))(anchor_layers[-1]))
                        predictor_sizes.append(class_layers[j]._keras_shape[1:3])

                    classes_concat = keras.layers.Concatenate(axis=1)(class_layers_reshape)
                    boxes_concat = keras.layers.Concatenate(axis=1)(box_layers_reshape)
                    anchors_concat = keras.layers.Concatenate(axis=1)(anchor_layers_reshape)
                    classes_softmax = keras.layers.Activation('softmax')(classes_concat)
                    box_preds = keras.layers.Concatenate(axis=2)([classes_softmax, boxes_concat, anchors_concat])

                    model = keras.models.Model(inputs=inputs, outputs=box_preds)
                else:
                    model = Model(inputs=inputs, outputs=outputs)
                model.summary()
                print(
                    "Note if testing compilation of a BBD model: a default set of model parameters was used during the test compilation.")
                stamp = datetime.datetime.fromtimestamp(time.time()).strftime('date_%Y_%m_%d_time_%H_%M_%S')
                cwd = os.getcwd()
                model_name = os.path.join(cwd, "temp/model_" + stamp + ".png")
                try:
                    plot_model(model, model_name, show_shapes=True)
                except:
                    pass
                if any(errors):
                    self.controller.home_menu.s_model_compile.set("It compiles w/o some layers")
                else:
                    self.controller.home_menu.s_model_compile.set("It compiles")
            except:
                self.controller.home_menu.s_model_compile.set("It could not compile")
        else:
            self.controller.home_menu.s_model_compile.set("No layers specified")
