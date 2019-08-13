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
"""src/engine/constructor.py

Constructs DLAE.
"""


from src.engine.configurations import EngineConfigurations
from src.engine.cnn import ConvolutionalNeuralNetwork
from src.engine.fcn import FullyConvolutionalNetwork
from src.engine.gan import GenerativeAdversairalNetwork
from src.engine.bbd import BoundingBoxDetector


class Dlae(object):
    def __init__(self, configs):
        self.configs = configs
        self.engine_configs = EngineConfigurations(self.configs)
        self.model = None
        self.errors = []
        self.get_model()

    def get_model(self):
        if self.engine_configs.dispatcher.model_signal == "CNN":
            self.model = ConvolutionalNeuralNetwork(self.engine_configs)

        elif self.engine_configs.dispatcher.model_signal == "FCN":
            self.model = FullyConvolutionalNetwork(self.engine_configs)

        elif self.engine_configs.dispatcher.model_signal == "GAN":
            self.model = GenerativeAdversairalNetwork(self.engine_configs)

        elif self.engine_configs.dispatcher.model_signal == "BBD":
            self.model = BoundingBoxDetector(self.engine_configs)

    def run(self):
        if self.engine_configs.dispatcher.type_signal == "Train":
            self.model.construct_graph()
            if any(self.model.errors):
                [self.errors.append(error) for error in self.model.errors]
            else:
                self.model.compile_graph()
                if any(self.model.errors):
                    [self.errors.append(error) for error in self.model.errors]
                else:
                    self.model.train_graph()
                    if any(self.model.errors):
                        [self.errors.append(error) for error in self.model.errors]

        elif self.engine_configs.dispatcher.type_signal == "Train from Checkpoint":
            self.model.retrain_graph()
            if any(self.model.errors):
                [self.errors.append(error) for error in self.model.errors]

        elif self.engine_configs.dispatcher.type_signal == "Inference":
            self.model.predict_on_graph()
            if any(self.model.errors):
                [self.errors.append(error) for error in self.model.errors]
