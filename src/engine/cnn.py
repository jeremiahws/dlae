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
"""src/engine/cnn.py

Constructs the convolutional neural network technique of DLAE.
"""


import tensorflow as tf
from src.utils.engine_utils import *
from src.utils.general_utils import write_hdf5
import datetime
import time


class ConvolutionalNeuralNetwork(object):
    def __init__(self, engine_configs):
        self.engine_configs = engine_configs
        self.graph = tf.get_default_graph()
        self.errors = []
        self.model = None
        self.parallel_model = None

    def construct_graph(self):
        if self.engine_configs.train_options.s_hardware == "cpu" or self.engine_configs.train_options.s_hardware == "multi-gpu":
            device = "/cpu:0"
        else:
            device = "/gpu:0"

        with tf.device(device):
            if any(self.engine_configs.layers.s_listOfLayers):
                inputs, outputs, _, errors = get_io(self.engine_configs.layers.s_listOfLayers)
                if any(errors):
                    [self.errors.append(error) for error in errors]
                else:
                    try:
                        self.model = keras.models.Model(inputs=inputs, outputs=outputs)
                    except:
                        self.errors.append('Level3Error:CouldNotConstructCnnGraph')

            else:
                self.errors.append('Level3Error:NoListofSerialLayersFound')

    def compile_graph(self):
        if self.engine_configs.train_options.i_nGpus > 1:
            self.parallel_model = ModelMGPU(self.model, self.engine_configs.train_options.i_nGpus)
            self.parallel_model.compile(optimizer=self.engine_configs.optimizer.optimizer,
                                        loss=self.engine_configs.loss_function.loss,
                                        metrics=self.engine_configs.monitors.monitors)
        else:
            self.model.compile(optimizer=self.engine_configs.optimizer.optimizer,
                               loss=self.engine_configs.loss_function.loss,
                               metrics=self.engine_configs.monitors.monitors)

    def train_graph(self):
        if self.engine_configs.val_data.val_generator is not None:
            val_data = self.engine_configs.val_data.val_generator.generate()
            val_steps = len(self.engine_configs.val_data.val_generator)
        else:
            val_data = None
            val_steps = None

        with self.graph.as_default():
            if self.engine_configs.train_options.i_nGpus > 1:
                self.parallel_model.fit_generator(generator=self.engine_configs.train_data.train_generator.generate(),
                                                  steps_per_epoch=len(self.engine_configs.train_data.train_generator),
                                                  epochs=self.engine_configs.train_options.i_epochs,
                                                  validation_data=val_data,
                                                  validation_steps=val_steps,
                                                  callbacks=self.engine_configs.callbacks.callbacks)
            else:
                self.model.fit_generator(generator=self.engine_configs.train_data.train_generator.generate(),
                                         steps_per_epoch=len(self.engine_configs.train_data.train_generator),
                                         epochs=self.engine_configs.train_options.i_epochs,
                                         validation_data=val_data,
                                         validation_steps=val_steps,
                                         callbacks=self.engine_configs.callbacks.callbacks)

        if self.engine_configs.saver.b_saveModel:
            self.model.save(self.engine_configs.saver.s_saveModelPath)

    def retrain_graph(self):
        pass

    def predict_on_graph(self):
        self.model = keras.models.load_model(self.engine_configs.loader.s_loadModelPath)
        predictions = self.model.predict_generator(self.engine_configs.test_data.test_generator.generate(),
                                                   steps=len(self.engine_configs.test_data.test_generator))

        stamp = datetime.datetime.fromtimestamp(time.time()).strftime('date_%Y_%m_%d_time_%H_%M_%S')

        write_hdf5('cnn_predictions_' + stamp + '.h5', predictions)

        del self.model
        K.clear_session()
