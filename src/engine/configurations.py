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
"""src/engine/configurations.py

Contains classes to construct the engine configurations from a configuration structure.
"""


import numpy as np
import keras
from ast import literal_eval
from src.utils.general_utils import str2bool, read_hdf5, read_bbd_X_Y, read_hdf5_multientry
from src.engine.loss_functions import TverskyLoss, SSDLoss


class EngineConfigurations(object):
    def __init__(self, configs):
        """
        Constructor class to build the engine configurations for DLAE.
        :param configs: engine configuration structure
        """
        self.dispatcher = Dispatcher(configs)
        self.data_preprocessing = Preprocessing(configs, self.dispatcher)
        self.train_data = TrainData(configs, self.data_preprocessing, self.dispatcher)
        self.val_data = ValidationData(configs, self.data_preprocessing, self.dispatcher)
        self.test_data = TestData(configs, self.data_preprocessing, self.dispatcher)
        self.learning_rate = LearningRate(configs)
        self.optimizer = Optimizer(configs, self.learning_rate)
        self.monitors = Monitors(configs)
        self.loader = Loader(configs)
        self.saver = Saver(configs)
        self.layers = Layers(configs)
        self.loss_function = LossFunction(configs, self.data_preprocessing)
        self.train_options = TrainingOptions(configs)
        self.callbacks = Callbacks(self.saver, self.learning_rate, self.train_options)
        self.augmentation = Augmentation(configs)


class Dispatcher(object):
    def __init__(self, configs):
        """
        Constructor class to build the dispatcher for DLAE.
        :param configs: engine configuration structure
        """
        self.model_signal = configs['config_file']['model_signal']
        self.type_signal = configs['config_file']['type_signal']


class Preprocessing(object):
    def __init__(self, configs, dispatcher):
        """
        Constructor class to build the image preprocessing configurations for DLAE.
        :param configs: engine configuration structure
        :param dispatcher: the engine dispatcher
        """
        self.f_minImageIntensity = configs['preprocessing']['minimum_image_intensity']
        self.f_maxImageIntensity = configs['preprocessing']['maximum_image_intensity']
        self.s_image_context = configs['preprocessing']['image_context']
        self.s_normalization_type = configs['preprocessing']['normalization_type']
        self.b_to_categorical = str2bool(configs['preprocessing']['categorical_switch'])
        self.b_weight_loss = str2bool(configs['preprocessing']['weight_loss_switch'])
        self.b_reshapeX = str2bool(configs['preprocessing']['reshape_X_switch'])
        self.t_reshapeX = configs['preprocessing']['reshape_X_dimensions']
        self.b_permuteX = str2bool(configs['preprocessing']['permute_X_switch'])
        self.t_permuteX = configs['preprocessing']['permute_X_dimensions']
        self.b_repeatX = str2bool(configs['preprocessing']['repeat_X_switch'])
        self.i_repeatX = configs['preprocessing']['repeat_X_quantity']
        self.b_reshapeY = str2bool(configs['preprocessing']['reshape_y_switch'])
        self.t_reshapeY = configs['preprocessing']['reshape_y_dimensions']
        self.b_permuteY = str2bool(configs['preprocessing']['permute_y_switch'])
        self.t_permuteY = configs['preprocessing']['permute_y_dimensions']

        if dispatcher.type_signal == "Train" or dispatcher.type_signal == "Train from Checkpoint":
            self.b_prepare_train = True
            self.b_prepare_val = True
            self.b_prepare_test = False

        elif dispatcher.type_signal == "Inference":
            self.b_prepare_train = False
            self.b_prepare_val = False
            self.b_prepare_test = True


class TrainData(object):
    def __init__(self, configs, preprocessing, dispatcher):
        """
        Constructor class to prepare the training data for DLAE.
        :param configs: engine configuration structure
        :param preprocessing: the preprocessing steps
        :param dispatcher: the engine dispatcher
        """
        self.errors = []
        self.warnings = []
        self.s_trainXPath = configs['paths']['train_X']
        self.s_trainYPath = configs['paths']['train_y']
        self.trainX = None
        self.trainY = None

        if (any(self.s_trainXPath) and any(self.s_trainYPath) is False)\
                or (any(self.s_trainYPath) and any(self.s_trainXPath) is False):
            self.errors.append('Level2Error:BothTrainImagesandAnnotationPathsNotSpecified')

        elif any(self.s_trainXPath) is False and any(self.s_trainYPath) is False:
            pass

        else:
            try:
                if dispatcher.model_signal == "BBD":
                    self.trainX, self.trainY = read_bbd_X_Y(self.s_trainXPath, self.s_trainYPath)
                else:
                    self.trainX = read_hdf5(self.s_trainXPath)
            except ImportError:
                self.errors.append('Level2Error:CouldNotLoadTrainXFile')

            try:
                if dispatcher.model_signal == "BBD":
                    pass
                else:
                    self.trainY = read_hdf5(self.s_trainYPath)
            except ImportError:
                self.errors.append('Level2Error:CouldNotLoadTrainYFile')

        if self.trainX is not None and self.trainY is not None\
                and preprocessing.b_prepare_train and dispatcher.model_signal != "BBD":
            try:
                if preprocessing.s_image_context == '2D':
                    if len(self.trainX.shape) == 2:
                        try:
                            self.trainX = np.expand_dims(self.trainX, axis=0)
                        except:
                            self.errors.append('Level2Error:CouldNotExpandTrainXDimensions')

                    if len(self.trainX.shape) == 3:
                        try:
                            self.trainX = np.expand_dims(self.trainX, axis=3)
                        except:
                            self.errors.append('Level2Error:CouldNotExpandTrainXDimensions')

                    if dispatcher.model_signal == "GAN" or dispatcher.model_signal == "FCN":
                        if len(self.trainY.shape) == 2:
                            try:
                                self.trainY = np.expand_dims(self.trainY, axis=0)
                            except:
                                self.errors.append('Level2Error:CouldNotExpandTrainYDimensions')

                        if len(self.trainY.shape) == 3:
                            try:
                                self.trainY = np.expand_dims(self.trainY, axis=3)
                            except:
                                self.errors.append('Level2Error:CouldNotExpandTrainYDimensions')

                elif preprocessing.s_image_context == '3D':
                    if len(self.trainX.shape) == 3:
                        try:
                            self.trainX = np.expand_dims(self.trainX, axis=0)
                        except:
                            self.errors.append('Level2Error:CouldNotExpandTrainXDimensions')

                    if len(self.trainX.shape) == 4:
                        try:
                            self.trainX = np.expand_dims(self.trainX, axis=4)
                        except:
                            self.errors.append('Level2Error:CouldNotExpandTrainXDimensions')

                    if dispatcher.model_signal == "GAN" or dispatcher.model_signal == "FCN":
                        if len(self.trainY.shape) == 3:
                            try:
                                self.trainY = np.expand_dims(self.trainY, axis=0)
                            except:
                                self.errors.append('Level2Error:CouldNotExpandTrainYDimensions')

                        if len(self.trainY.shape) == 4:
                            try:
                                self.trainY = np.expand_dims(self.trainY, axis=4)
                            except:
                                self.errors.append('Level2Error:CouldNotExpandTrainYDimensions')

                if preprocessing.b_permuteX:
                    try:
                        if any(preprocessing.t_permuteX) is False:
                            self.errors.append('Level2Error:NeedToSpecifyTrainXPermutationDimensions')

                        else:
                            self.trainX = np.transpose(self.trainX, axes=literal_eval(preprocessing.t_permuteX))

                    except SyntaxError:
                        self.errors.append('Level2Error:CouldNotPerformSpecifiedTrainXPermutation')

                if preprocessing.b_reshapeX:
                    try:
                        if any(preprocessing.t_reshapeX) is False:
                            self.errors.append('Level2Error:NeedToSpecifyTrainXReshapeDimensions')

                        else:
                            self.trainX = np.reshape(self.trainX, newshape=literal_eval(preprocessing.t_reshapeX))

                    except SyntaxError:
                        self.errors.append('Level2Error:CouldNotPerformSpecifiedTrainXReshape')

                if preprocessing.b_permuteY:
                    try:
                        if any(preprocessing.t_permuteY) is False:
                            self.errors.append('Level2Error:NeedToSpecifyTrainYPermutationDimensions')

                        else:
                            self.trainY = np.transpose(self.trainY, axes=literal_eval(preprocessing.t_permuteY))

                    except SyntaxError:
                        self.errors.append('Level2Error:CouldNotPerformSpecifiedTrainYPermutation')

                if preprocessing.b_reshapeY:
                    try:
                        if any(preprocessing.t_reshapeY) is False:
                            self.errors.append('Level2Error:NeedToSpecifyTrainYReshapeDimensions')

                        else:
                            self.trainY = np.reshape(self.trainY, newshape=literal_eval(preprocessing.t_reshapeY))

                    except SyntaxError:
                        self.errors.append('Level2Error:CouldNotPerformSpecifiedTrainYReshape')

                if self.trainX.shape[0] != self.trainY.shape[0]:
                    self.errors.append('Level2Error:NumberofTrainXImagesMustEqualNumberofTrainYAnnotations')

                if preprocessing.s_normalization_type == 'none':
                    pass

                elif preprocessing.s_normalization_type == 'X from [0, 1]':
                    if (any(preprocessing.f_minImageIntensity) is False)\
                            or (any(preprocessing.f_maxImageIntensity) is False)\
                            or (any(preprocessing.f_minImageIntensity) is False and any(preprocessing.f_maxImageIntensity) is False):
                        self.errors.append('Level2Error:SetBothMinandMaxImageIntensityforImageNormalization')

                    else:
                        try:
                            minimum = float(preprocessing.f_minImageIntensity)
                            maximum = float(preprocessing.f_maxImageIntensity)
                            self.trainX = (self.trainX - minimum) / (maximum - minimum)
                        except:
                            self.errors.append('Level2Error:CouldNotNormalizeTrainXData')

                elif preprocessing.s_normalization_type == 'X from [-1, 1]':
                    if (any(preprocessing.f_minImageIntensity) is False)\
                            or (any(preprocessing.f_maxImageIntensity) is False)\
                            or (any(preprocessing.f_minImageIntensity) is False and any(preprocessing.f_maxImageIntensity) is False):
                        self.errors.append('Level2Error:SetBothMinandMaxImageIntensityforImageNormalization')

                    else:
                        try:
                            minimum = float(preprocessing.f_minImageIntensity)
                            maximum = float(preprocessing.f_maxImageIntensity)
                            self.trainX = 2 * (((self.trainX - minimum) / (maximum - minimum)) - 0.5)
                        except:
                            self.errors.append('Level2Error:CouldNotNormalizeTrainXData')

                elif preprocessing.s_normalization_type == 'X, Y from [0, 1]':
                    if (any(preprocessing.f_minImageIntensity) is False)\
                            or (any(preprocessing.f_maxImageIntensity) is False)\
                            or (any(preprocessing.f_minImageIntensity) is False and any(preprocessing.f_maxImageIntensity) is False):
                        self.errors.append('Level2Error:SetBothMinandMaxImageIntensityforImageNormalization')

                    else:
                        try:
                            minimums = preprocessing.f_minImageIntensity.split(',')
                            maximums = preprocessing.f_maxImageIntensity.split(',')
                            self.trainX = (self.trainX - float(minimums[0])) / (float(maximums[0]) - float(minimums[0]))
                            self.trainY = (self.trainY - float(minimums[1])) / (float(maximums[1]) - float(minimums[1]))
                        except:
                            self.errors.append('Level2Error:CouldNotNormalizeTrainXAndTrainYData')

                elif preprocessing.s_normalization_type == 'X, Y from [-1, 1]':
                    if (any(preprocessing.f_minImageIntensity) is False)\
                            or (any(preprocessing.f_maxImageIntensity) is False)\
                            or (any(preprocessing.f_minImageIntensity) is False and any(preprocessing.f_maxImageIntensity) is False):
                        self.errors.append('Level2Error:SetBothMinandMaxImageIntensityforImageNormalization')

                    else:
                        try:
                            minimums = preprocessing.f_minImageIntensity.split(',')
                            maximums = preprocessing.f_maxImageIntensity.split(',')
                            self.trainX = 2 * (((self.trainX - float(minimums[0])) / (float(maximums[0]) - float(minimums[0]))) - 0.5)
                            self.trainY = 2 * (((self.trainY - float(minimums[1])) / (float(maximums[1]) - float(minimums[1]))) - 0.5)
                        except:
                            self.errors.append('Level2Error:CouldNotNormalizeTrainXAndTrainYData')

                if preprocessing.i_repeatX:
                    try:
                        self.trainX = np.repeat(self.trainX, repeats= int(preprocessing.i_repeatX),
                                                axis=np.ndim(self.trainX) - 1)
                    except:
                        self.errors.append('Level2Error:CouldNotRepeatTrainXAlongChannels')

            except SyntaxError:
                self.errors.append('Level2Error:CouldNotPerformPreprocessingonTrainData')


class ValidationData(object):
    def __init__(self, configs, preprocessing, dispatcher):
        """
        Constructor class to prepare the validation data for DLAE.
        :param configs: engine configuration structure
        :param preprocessing: the preprocessing steps
        :param dispatcher: the engine dispatcher
        """
        self.errors = []
        self.warnings = []
        self.s_valXPath = configs['paths']['validation_X']
        self.s_valYPath = configs['paths']['validation_y']
        self.valX = None
        self.valY = None

        if (any(self.s_valXPath) and any(self.s_valYPath) is False)\
                or (any(self.s_valYPath) and any(self.s_valXPath) is False):
            self.errors.append('Level2Error:BothValidationImagesandAnnotationPathsNotSpecified')

        elif any(self.s_valXPath) is False and any(self.s_valYPath) is False:
            pass

        else:
            try:
                if dispatcher.model_signal == "BBD":
                    self.valX, self.valY = read_bbd_X_Y(self.s_valXPath, self.s_valYPath)
                else:
                    self.valX = read_hdf5(self.s_valXPath)
            except ImportError:
                self.errors.append('Level2Error:CouldNotLoadValidationXFile')

            try:
                if dispatcher.model_signal == "BBD":
                    pass
                else:
                    self.valY = read_hdf5(self.s_valYPath)
            except ImportError:
                self.errors.append('Level2Error:CouldNotLoadValidationYFile')

        if self.valX is not None and self.valY is not None\
                and preprocessing.b_prepare_val and dispatcher.model_signal != "BBD":
            try:
                if preprocessing.s_image_context == '2D':
                    if len(self.valX.shape) == 2:
                        try:
                            self.valX = np.expand_dims(self.valX, axis=0)
                        except:
                            self.errors.append('Level2Error:CouldNotExpandValidationXDimensions')

                    if len(self.valX.shape) == 3:
                        try:
                            self.valX = np.expand_dims(self.valX, axis=3)
                        except:
                            self.errors.append('Level2Error:CouldNotExpandValidationXDimensions')

                    if dispatcher.model_signal == "GAN" or dispatcher.model_signal == "FCN":
                        if len(self.valY.shape) == 2:
                            try:
                                self.valY = np.expand_dims(self.valY, axis=0)
                            except:
                                self.errors.append('Level2Error:CouldNotExpandValidationYDimensions')

                        if len(self.valY.shape) == 3:
                            try:
                                self.valY = np.expand_dims(self.valY, axis=3)
                            except:
                                self.errors.append('Level2Error:CouldNotExpandValidationYDimensions')

                elif preprocessing.s_image_context == '3D':
                    if len(self.valX.shape) == 3:
                        try:
                            self.valX = np.expand_dims(self.valX, axis=0)
                        except:
                            self.errors.append('Level2Error:CouldNotExpandValidationXDimensions')

                    if len(self.valX.shape) == 4:
                        try:
                            self.valX = np.expand_dims(self.valX, axis=4)
                        except:
                            self.errors.append('Level2Error:CouldNotExpandValidationXDimensions')

                    if dispatcher.model_signal == "GAN" or dispatcher.model_signal == "FCN":
                        if len(self.valY.shape) == 3:
                            try:
                                self.valY = np.expand_dims(self.valY, axis=0)
                            except:
                                self.errors.append('Level2Error:CouldNotExpandValidationYDimensions')

                        if len(self.valY.shape) == 4:
                            try:
                                self.valY = np.expand_dims(self.valY, axis=4)
                            except:
                                self.errors.append('Level2Error:CouldNotExpandValidationYDimensions')

                if self.valX.shape[0] != self.valY.shape[0]:
                    self.errors.append('Level2Error:NumberofValidationXImagesMustEqualNumberofValidationYAnnotations')

                if preprocessing.b_permuteX:
                    try:
                        if any(preprocessing.t_permuteX) is False:
                            self.errors.append('Level2Error:NeedToSpecifyValidationXPermutationDimensions')

                        else:
                            self.valX = np.transpose(self.valX, axes=literal_eval(preprocessing.t_permuteX))

                    except SyntaxError:
                        self.errors.append('Level2Error:CouldNotPerformSpecifiedValidationXPermutation')

                if preprocessing.b_reshapeX:
                    try:
                        if any(preprocessing.t_reshapeX) is False:
                            self.errors.append('Level2Error:NeedToSpecifyValidationXReshapeDimensions')

                        else:
                            self.valX = np.reshape(self.valX, newshape=literal_eval(preprocessing.t_reshapeX))

                    except SyntaxError:
                        self.errors.append('Level2Error:CouldNotPerformSpecifiedValidationXReshape')

                if preprocessing.b_permuteY:
                    try:
                        if any(preprocessing.t_permuteY) is False:
                            self.errors.append('Level2Error:NeedToSpecifyValidationYPermutationDimensions')

                        else:
                            self.valY = np.transpose(self.valY, axes=literal_eval(preprocessing.t_permuteY))

                    except SyntaxError:
                        self.errors.append('Level2Error:CouldNotPerformSpecifiedValidationYPermutation')

                if preprocessing.b_reshapeY:
                    try:
                        if any(preprocessing.t_reshapeY) is False:
                            self.errors.append('Level2Error:NeedToSpecifyValidationYReshapeDimensions')

                        else:
                            self.valY = np.reshape(self.valY, newshape=literal_eval(preprocessing.t_reshapeY))

                    except SyntaxError:
                        self.errors.append('Level2Error:CouldNotPerformSpecifiedValidationYReshape')

                if preprocessing.s_normalization_type == 'none':
                    pass

                elif preprocessing.s_normalization_type == 'X from [0, 1]':
                    if (any(preprocessing.f_minImageIntensity) is False)\
                            or (any(preprocessing.f_maxImageIntensity) is False)\
                            or (any(preprocessing.f_minImageIntensity) is False and any(preprocessing.f_maxImageIntensity) is False):
                        self.errors.append('Level2Error:SetBothMinandMaxImageIntensityforImageNormalization')

                    else:
                        try:
                            minimum = float(preprocessing.f_minImageIntensity)
                            maximum = float(preprocessing.f_maxImageIntensity)
                            self.valX = (self.valX - minimum) / (maximum - minimum)
                        except:
                            self.errors.append('Level2Error:CouldNotNormalizeValidationXData')

                elif preprocessing.s_normalization_type == 'X from [-1, 1]':
                    if (any(preprocessing.f_minImageIntensity) is False)\
                            or (any(preprocessing.f_maxImageIntensity) is False)\
                            or (any(preprocessing.f_minImageIntensity) is False and any(preprocessing.f_maxImageIntensity) is False):
                        self.errors.append('Level2Error:SetBothMinandMaxImageIntensityforImageNormalization')

                    else:
                        try:
                            minimum = float(preprocessing.f_minImageIntensity)
                            maximum = float(preprocessing.f_maxImageIntensity)
                            self.valX = 2 * (((self.valX - minimum) / (maximum - minimum)) - 0.5)
                        except:
                            self.errors.append('Level2Error:CouldNotNormalizeValidationXData')

                elif preprocessing.s_normalization_type == 'X, Y from [0, 1]':
                    if (any(preprocessing.f_minImageIntensity) is False)\
                            or (any(preprocessing.f_maxImageIntensity) is False)\
                            or (any(preprocessing.f_minImageIntensity) is False and any(preprocessing.f_maxImageIntensity) is False):
                        self.errors.append('Level2Error:SetBothMinandMaxImageIntensityforImageNormalization')

                    else:
                        try:
                            minimums = preprocessing.f_minImageIntensity.split(',')
                            maximums = preprocessing.f_maxImageIntensity.split(',')
                            self.valX = (self.valX - float(minimums[0])) / (float(maximums[0]) - float(minimums[0]))
                            self.valY = (self.valY - float(minimums[1])) / (float(maximums[1]) - float(minimums[1]))
                        except:
                            self.errors.append('Level2Error:CouldNotNormalizeValidationXAndValidationYData')

                elif preprocessing.s_normalization_type == 'X, Y from [-1, 1]':
                    if (any(preprocessing.f_minImageIntensity) is False)\
                            or (any(preprocessing.f_maxImageIntensity) is False)\
                            or (any(preprocessing.f_minImageIntensity) is False and any(preprocessing.f_maxImageIntensity) is False):
                        self.errors.append('Level2Error:SetBothMinandMaxImageIntensityforImageNormalization')

                    else:
                        try:
                            minimums = preprocessing.f_minImageIntensity.split(',')
                            maximums = preprocessing.f_maxImageIntensity.split(',')
                            self.valX = 2 * (((self.valX - float(minimums[0])) / (float(maximums[0]) - float(minimums[0]))) - 0.5)
                            self.valY = 2 * (((self.valY - float(minimums[1])) / (float(maximums[1]) - float(minimums[1]))) - 0.5)
                        except:
                            self.errors.append('Level2Error:CouldNotNormalizeValidationXAndValidationYData')

                if preprocessing.i_repeatX:
                    try:
                        self.valX = np.repeat(self.valX, repeats=int(preprocessing.i_repeatX),
                                              axis=np.ndim(self.valX) - 1)
                    except:
                        self.errors.append('Level2Error:CouldNotRepeatValidationXAlongChannels')

            except SyntaxError:
                self.errors.append('Level2Error:CouldNotPerformPreprocessingonValidationData')


class TestData(object):
    def __init__(self, configs, preprocessing, dispatcher):
        """
        Constructor class to prepare the testing data for DLAE.
        :param configs: engine configuration structure
        :param preprocessing: the preprocessing steps
        :param dispatcher: the engine dispatcher
        """
        self.errors = []
        self.warnings = []
        self.s_testXPath = configs['paths']['test_X']
        self.testX = None

        if any(self.s_testXPath) is False:
            pass
        else:
            try:
                if dispatcher.model_signal == "BBD":
                    self.testX = read_hdf5_multientry(self.s_testXPath)
                else:
                    self.testX = read_hdf5(self.s_testXPath)
            except ImportError:
                self.errors.append('Level2Error:CouldNotLoadTestXFile')

        if self.testX is not None and preprocessing.b_prepare_test and dispatcher.model_signal != "BBD":
            try:
                if preprocessing.s_image_context == '2D':
                    if len(self.testX.shape) == 2:
                        try:
                            self.testX = np.expand_dims(self.testX, axis=0)
                        except:
                            self.errors.append('Level2Error:CouldNotExpandTestXDimensions')

                    if len(self.testX.shape) == 3:
                        try:
                            self.testX = np.expand_dims(self.testX, axis=3)
                        except:
                            self.errors.append('Level2Error:CouldNotExpandTestXDimensions')

                elif preprocessing.s_image_context == '3D':
                    if len(self.testX.shape) == 3:
                        try:
                            self.testX = np.expand_dims(self.testX, axis=0)
                        except:
                            self.errors.append('Level2Error:CouldNotExpandTestXDimensions')

                    if len(self.testX.shape) == 4:
                        try:
                            self.testX = np.expand_dims(self.testX, axis=4)
                        except:
                            self.errors.append('Level2Error:CouldNotExpandTestXDimensions')

                if preprocessing.b_permuteX:
                    try:
                        if any(preprocessing.t_permuteX) is False:
                            self.errors.append('Level2Error:NeedToSpecifyTestXPermutationDimensions')

                        else:
                            self.testX = np.transpose(self.testX, axes=literal_eval(preprocessing.t_permuteX))

                    except SyntaxError:
                        self.errors.append('Level2Error:CouldNotPerformSpecifiedTestXPermutation')

                if preprocessing.b_reshapeX:
                    try:
                        if any(preprocessing.t_reshapeX) is False:
                            self.errors.append('Level2Error:NeedToSpecifyTestXReshapeDimensions')

                        else:
                            self.testX = np.reshape(self.testX, newshape=literal_eval(preprocessing.t_reshapeX))

                    except SyntaxError:
                        self.errors.append('Level2Error:CouldNotPerformSpecifiedTestXReshape')

                if preprocessing.s_normalization_type == 'none':
                    pass

                elif preprocessing.s_normalization_type == 'X from [0, 1]':
                    if (any(preprocessing.f_minImageIntensity) is False)\
                            or (any(preprocessing.f_maxImageIntensity) is False)\
                            or (any(preprocessing.f_minImageIntensity) is False and any(preprocessing.f_maxImageIntensity) is False):
                        self.errors.append('Level2Error:SetBothMinandMaxImageIntensityforImageNormalization')

                    else:
                        try:
                            minimum = float(preprocessing.f_minImageIntensity)
                            maximum = float(preprocessing.f_maxImageIntensity)
                            self.testX = (self.testX - minimum) / (maximum - minimum)
                        except:
                            self.errors.append('Level2Error:CouldNotNormalizeTestXData')

                elif preprocessing.s_normalization_type == 'X from [-1, 1]':
                    if (any(preprocessing.f_minImageIntensity) is False)\
                            or (any(preprocessing.f_maxImageIntensity) is False)\
                            or (any(preprocessing.f_minImageIntensity) is False and any(preprocessing.f_maxImageIntensity) is False):
                        self.errors.append('Level2Error:SetBothMinandMaxImageIntensityforImageNormalization')

                    else:
                        try:
                            minimum = float(preprocessing.f_minImageIntensity)
                            maximum = float(preprocessing.f_maxImageIntensity)
                            self.testX = 2 * (((self.testX - minimum) / (maximum - minimum)) - 0.5)
                        except:
                            self.errors.append('Level2Error:CouldNotNormalizeTestXData')

                elif preprocessing.s_normalization_type == 'X, Y from [0, 1]':
                    if (any(preprocessing.f_minImageIntensity) is False)\
                            or (any(preprocessing.f_maxImageIntensity) is False)\
                            or (any(preprocessing.f_minImageIntensity) is False and any(preprocessing.f_maxImageIntensity) is False):
                        self.errors.append('Level2Error:SetBothMinandMaxImageIntensityforImageNormalization')

                    else:
                        try:
                            minimums = preprocessing.f_minImageIntensity.split(',')
                            maximums = preprocessing.f_maxImageIntensity.split(',')
                            self.testX = (self.testX - float(minimums[0])) / (float(maximums[0]) - float(minimums[0]))
                        except:
                            self.errors.append('Level2Error:CouldNotNormalizeTestXData')

                elif preprocessing.s_normalization_type == 'X, Y from [-1, 1]':
                    if (any(preprocessing.f_minImageIntensity) is False)\
                            or (any(preprocessing.f_maxImageIntensity) is False)\
                            or (any(preprocessing.f_minImageIntensity) is False and any(preprocessing.f_maxImageIntensity) is False):
                        self.errors.append('Level2Error:SetBothMinandMaxImageIntensityforImageNormalization')

                    else:
                        try:
                            minimums = preprocessing.f_minImageIntensity.split(',')
                            maximums = preprocessing.f_maxImageIntensity.split(',')
                            self.testX = 2 * (((self.testX - float(minimums[0])) / (float(maximums[0]) - float(minimums[0]))) - 0.5)
                        except:
                            self.errors.append('Level2Error:CouldNotNormalizeTestXData')

                if preprocessing.i_repeatX:
                    try:
                        self.testX = np.repeat(self.testX, repeats=int(preprocessing.i_repeatX),
                                               axis=np.ndim(self.testX) - 1)
                    except:
                        self.errors.append('Level2Error:CouldNotRepeatTestXAlongChannels')

            except SyntaxError:
                self.errors.append('Level2Error:CouldNotPerformPreprocessingonTestData')


class LearningRate(object):
    def __init__(self, configs):
        """
        Constructor class to prepare the learning rate configurations for DLAE.
        :param configs: engine configuration structure
        """
        self.f_lr = float(configs['learning_rate_schedule']['learning_rate'])
        self.f_lrDecay = float(configs['learning_rate_schedule']['learning_rate_decay_factor'])
        self.b_lrDecayOnPlateau = str2bool(configs['learning_rate_schedule']['decay_on_plateau_switch'])
        self.f_lrDecayOnPlateauFactor = float(configs['learning_rate_schedule']['decay_on_plateau_factor'])
        self.i_lrDecayOnPlateauPatience = int(configs['learning_rate_schedule']['decay_on_plateau_patience'])
        self.b_lrStepDecay = str2bool(configs['learning_rate_schedule']['step_decay_switch'])
        self.f_lrStepDecayFactor = float(configs['learning_rate_schedule']['step_decay_factor'])
        self.i_lrStepDecayPeriod = int(configs['learning_rate_schedule']['step_decay_period'])
        self.s_lrDiscriminator = configs['learning_rate_schedule']['discriminator_learning_rate']
        self.s_lrGAN = configs['learning_rate_schedule']['gan_learning_rate']


class Optimizer(object):
    def __init__(self, configs, learning_rate):
        """
        Constructor class to prepare the optimizer configurations for DLAE.
        :param configs: engine configuration structure
        :param learning_rate: the learning rate configurations
        """
        self.s_optimizer = configs['optimizer']['optimizer']
        self.s_d_optimizer = configs['optimizer']['discriminator_optimizer']
        self.s_gan_optimizer = configs['optimizer']['gan_optimizer']
        self.f_optimizerBeta1 = float(configs['optimizer']['beta1'])
        self.f_optimizerBeta2 = float(configs['optimizer']['beta2'])
        self.f_optimizerEpsilon = literal_eval(configs['optimizer']['epsilon'])
        self.f_optimizerRho = float(configs['optimizer']['rho'])
        self.f_optimizerMomentum = float(configs['optimizer']['momentum'])
        self.learning_rate = learning_rate
        self.set_optimizer()
        self.set_d_optimizer()
        self.set_gan_optimizer()

    def set_optimizer(self):
        if self.s_optimizer == 'Adam':
            self.optimizer = keras.optimizers.Adam(lr=self.learning_rate.f_lr, beta_1=self.f_optimizerBeta1,
                                                   beta_2=self.f_optimizerBeta2, epsilon=self.f_optimizerEpsilon,
                                                   decay=self.learning_rate.f_lrDecay)

        if self.s_optimizer == 'NAdam':
            self.optimizer = keras.optimizers.Nadam(lr=self.learning_rate.f_lr, beta_1=self.f_optimizerBeta1,
                                                    beta_2=self.f_optimizerBeta2, epsilon=self.f_optimizerEpsilon,
                                                    decay=self.learning_rate.f_lrDecay)

        elif self.s_optimizer == 'SGD':
            self.optimizer = keras.optimizers.SGD(lr=self.learning_rate.f_lr, momentum=self.f_optimizerMomentum,
                                                  decay=self.learning_rate.f_lrDecay)

        elif self.s_optimizer == 'RMSprop':
            self.optimizer = keras.optimizers.RMSprop(lr=self.learning_rate.f_lr, rho=self.f_optimizerRho,
                                                      epsilon=self.f_optimizerEpsilon,
                                                      decay=self.learning_rate.f_lrDecay)

        elif self.s_optimizer == 'Adagrad':
            self.optimizer = keras.optimizers.Adagrad(lr=self.learning_rate.f_lr, epsilon=self.f_optimizerEpsilon,
                                                      decay=self.learning_rate.f_lrDecay)

        elif self.s_optimizer == 'Adadelta':
            self.optimizer = keras.optimizers.Adadelta(lr=self.learning_rate.f_lr, rho=self.f_optimizerRho,
                                                       epsilon=self.f_optimizerEpsilon,
                                                       decay=self.learning_rate.f_lrDecay)

        elif self.s_optimizer == 'Adamax':
            self.optimizer = keras.optimizers.Adamax(lr=self.learning_rate.f_lr, beta_1=self.f_optimizerBeta1,
                                                     beta_2=self.f_optimizerBeta2, epsilon=self.f_optimizerEpsilon,
                                                     decay=self.learning_rate.f_lrDecay)

    def set_d_optimizer(self):
        d_optimizer = self.s_d_optimizer.split(':')
        d_lr = self.learning_rate.s_lrDiscriminator.split(':')
        if self.s_optimizer == 'Adam':
            self.d_optimizer = keras.optimizers.Adam(lr=float(d_lr[0]), beta_1=float(d_optimizer[1]),
                                                     beta_2=float(d_optimizer[2]), epsilon=literal_eval(d_optimizer[5]),
                                                     decay=float(d_lr[1]))

        elif self.s_optimizer == 'NAdam':
            self.d_optimizer = keras.optimizers.Nadam(lr=float(d_lr[0]), beta_1=float(d_optimizer[1]),
                                                      beta_2=float(d_optimizer[2]),
                                                      epsilon=literal_eval(d_optimizer[5]), decay=float(d_lr[1]))

        elif self.s_optimizer == 'SGD':
            self.d_optimizer = keras.optimizers.SGD(lr=float(d_lr[0]), momentum=float(d_optimizer[4]),
                                                    decay=float(d_lr[1]))

        elif self.s_optimizer == 'RMSprop':
            self.d_optimizer = keras.optimizers.RMSprop(lr=float(d_lr[0]), rho=float(d_optimizer[3]),
                                                        epsilon=literal_eval(d_optimizer[5]), decay=float(d_lr[1]))

        elif self.s_optimizer == 'Adagrad':
            self.d_optimizer = keras.optimizers.Adagrad(lr=float(d_lr[0]), epsilon=literal_eval(d_optimizer[5]),
                                                        decay=float(d_lr[1]))

        elif self.s_optimizer == 'Adadelta':
            self.d_optimizer = keras.optimizers.Adadelta(lr=float(d_lr[0]), rho=float(d_optimizer[3]),
                                                         epsilon=literal_eval(d_optimizer[5]), decay=float(d_lr[1]))

        elif self.s_optimizer == 'Adamax':
            self.d_optimizer = keras.optimizers.Adamax(lr=float(d_lr[0]), beta_1=float(d_optimizer[1]),
                                                       beta_2=float(d_optimizer[2]),
                                                       epsilon=literal_eval(d_optimizer[5]), decay=float(d_lr[1]))

    def set_gan_optimizer(self):
        gan_optimizer = self.s_gan_optimizer.split(':')
        gan_lr = self.learning_rate.s_lrGAN.split(':')
        if self.s_optimizer == 'Adam':
            self.gan_optimizer = keras.optimizers.Adam(lr=float(gan_lr[0]), beta_1=float(gan_optimizer[1]),
                                                       beta_2=float(gan_optimizer[2]),
                                                       epsilon=literal_eval(gan_optimizer[5]), decay=float(gan_lr[1]))

        elif self.s_optimizer == 'NAdam':
            self.gan_optimizer = keras.optimizers.Nadam(lr=float(gan_lr[0]), beta_1=float(gan_optimizer[1]),
                                                        beta_2=float(gan_optimizer[2]),
                                                        epsilon=literal_eval(gan_optimizer[5]), decay=float(gan_lr[1]))

        elif self.s_optimizer == 'SGD':
            self.gan_optimizer = keras.optimizers.SGD(lr=float(gan_lr[0]), momentum=float(gan_optimizer[4]),
                                                      decay=float(gan_lr[1]))

        elif self.s_optimizer == 'RMSprop':
            self.gan_optimizer = keras.optimizers.RMSprop(lr=float(gan_lr[0]), rho=float(gan_optimizer[3]),
                                                          epsilon=literal_eval(gan_optimizer[5]),
                                                          decay=float(gan_lr[1]))

        elif self.s_optimizer == 'Adagrad':
            self.gan_optimizer = keras.optimizers.Adagrad(lr=float(gan_lr[0]), epsilon=literal_eval(gan_optimizer[5]),
                                                          decay=float(gan_lr[1]))

        elif self.s_optimizer == 'Adadelta':
            self.gan_optimizer = keras.optimizers.Adadelta(lr=float(gan_lr[0]), rho=float(gan_optimizer[3]),
                                                           epsilon=literal_eval(gan_optimizer[5]),
                                                           decay=float(gan_lr[1]))

        elif self.s_optimizer == 'Adamax':
            self.gan_optimizer = keras.optimizers.Adamax(lr=float(gan_lr[0]), beta_1=float(gan_optimizer[1]),
                                                         beta_2=float(gan_optimizer[2]),
                                                         epsilon=literal_eval(gan_optimizer[5]), decay=float(gan_lr[1]))


class Monitors(object):
    def __init__(self, configs):
        """
        Constructor class to prepare the monitor configurations for DLAE.
        :param configs: engine configuration structure
        """
        self.b_monitorMSE = str2bool(configs['monitors']['mse_switch'])
        self.b_monitorMAE = str2bool(configs['monitors']['mae_switch'])
        self.b_monitorAcc = str2bool(configs['monitors']['accuracy_switch'])
        self.monitors = []

        if self.b_monitorMSE:
            self.monitors.append(keras.metrics.mse)

        if self.b_monitorMAE:
            self.monitors.append(keras.metrics.mae)

        if self.b_monitorAcc:
            self.monitors.append(keras.metrics.categorical_accuracy)


class Loader(object):
    def __init__(self, configs):
        """
        Constructor class to prepare the loading configurations for DLAE.
        :param configs: engine configuration structure
        """
        self.s_loadModelPath = configs['paths']['load_model']
        self.s_loadCheckpointPath = configs['paths']['load_checkpoint']


class Saver(object):
    def __init__(self, configs):
        """
        Constructor class to prepare the saving configurations for DLAE.
        :param configs: engine configuration structure
        """
        self.errors = []
        self.warnings = []
        self.b_saveModel = str2bool(configs['save_configurations']['save_model_switch'])
        self.s_saveModelPath = configs['save_configurations']['save_model_path']

        if self.b_saveModel and any(self.s_saveModelPath) is False:
            self.errors.append("Level2Error:NoSaveModelPathSpecified")

        self.b_saveCSV = str2bool(configs['save_configurations']['save_csv_switch'])
        self.s_saveCSVPath = configs['save_configurations']['save_csv_path']

        if self.b_saveCSV and any(self.s_saveCSVPath) is False:
            self.errors.append("Level2Error:NoSaveCSVPathSpecified")

        self.b_saveCkpt = str2bool(configs['save_configurations']['save_checkpoints_switch'])
        self.s_saveCkptPath = configs['save_configurations']['save_checkpoints_path']

        if self.b_saveCkpt and any(self.s_saveCkptPath) is False:
            self.errors.append("Level2Error:NoSaveCheckpointPathSpecified")

        self.i_saveCkptFrequency = int(configs['save_configurations']['save_checkpoints_frequency'])
        self.b_recordTensorboardLogs = str2bool(configs['save_configurations']['save_tensorboard_switch'])
        self.s_recordTensorboardPath = configs['save_configurations']['save_tensorboard_path']

        if self.b_recordTensorboardLogs and any(self.s_recordTensorboardPath) is False:
            self.errors.append("Level2Error:NoSaveTensorboardPathSpecified")

        self.i_recordTensorboardFrequency = int(configs['save_configurations']['save_tensorboard_frequency'])


class Layers(object):
    def __init__(self, configs):
        """
        Constructor class to store the layer configurations for DLAE.
        :param configs: engine configuration structure
        """
        self.s_listOfLayers = configs['layers']['serial_layer_list']
        self.s_listOfGeneratorLayers = configs['layers']['generator_layer_list']
        self.s_listOfDiscriminatorLayers = configs['layers']['discriminator_layer_list']
        self.t_input_shape = literal_eval(configs['config_file']['input_shape'])


class LossFunction(object):
    def __init__(self, configs, preprocessing):
        """
        Constructor class to prepare the loss function configurations for DLAE.
        :param configs: engine configuration structure
        """
        self.s_lossFunction = configs['loss_function']['loss']
        self.f_parameter1 = float(configs['loss_function']['parameter1'])
        self.f_parameter2 = float(configs['loss_function']['parameter2'])
        self.image_context = preprocessing.s_image_context
        self.loss = None
        self.set_loss_function()

    def set_loss_function(self):
        if self.s_lossFunction == 'categorical_crossentropy':
            self.loss = keras.losses.categorical_crossentropy

        elif self.s_lossFunction == 'sparse_categorical_crossentropy':
            self.loss = keras.losses.sparse_categorical_crossentropy

        elif self.s_lossFunction == 'mean_squared_error':
            self.loss = keras.losses.mean_squared_error

        elif self.s_lossFunction == 'mean_absolute_error':
            self.loss = keras.losses.mean_absolute_error

        elif self.s_lossFunction == 'tversky':
            tversky_loss = TverskyLoss(alpha=self.f_parameter1, beta=self.f_parameter2)
            if self.image_context == "2D":
                self.loss = tversky_loss.compute_loss_2D
            else:
                self.loss = tversky_loss.compute_loss_3D

        elif self.s_lossFunction == 'pix2pix':
            self.loss = "pix2pix"

        elif self.s_lossFunction == 'cyclegan':
            self.loss = "cyclegan"

        elif self.s_lossFunction == 'ssd':
            ssd_loss = SSDLoss(neg_pos_ratio=self.f_parameter1, alpha=self.f_parameter2)
            self.loss = ssd_loss.compute_loss


class TrainingOptions(object):
    def __init__(self, configs):
        """
        Constructor class to prepare the training options for DLAE.
        :param configs: engine configuration structure
        """
        self.i_batchSize = int(configs['training_configurations']['batch_size'])
        self.i_epochs = int(configs['training_configurations']['epochs'])
        self.s_hardware = configs['training_configurations']['hardware']
        self.i_nGpus = int(configs['training_configurations']['number_of_gpus'])
        self.b_shuffleData = str2bool(configs['training_configurations']['shuffle_data_switch'])
        self.f_validationSplit = float(configs['training_configurations']['validation_split'])
        self.b_earlyStop = str2bool(configs['training_configurations']['early_stop_switch'])
        self.i_earlyStopPatience = int(configs['training_configurations']['early_stop_patience'])
        self.s_scalingType = configs['bbd_options']['scaling_type']
        self.l_scales = list(literal_eval(configs['bbd_options']['scales']))
        self.s_aspectRatiosType = configs['bbd_options']['aspect_ratios_type']
        self.l_aspectRatios = literal_eval(configs['bbd_options']['aspect_ratios'])
        self.i_numberOfBBDClasses = int(configs['bbd_options']['number_classes'])
        self.l_steps = literal_eval(configs['bbd_options']['steps'])
        self.l_offsets = literal_eval(configs['bbd_options']['offsets'])
        self.l_variances = literal_eval(configs['bbd_options']['variances'])
        self.f_confidenceThreshold = float(configs['bbd_options']['confidence_threshold'])
        self.f_iouThreshold = float(configs['bbd_options']['iou_threshold'])
        self.f_posIouThreshold = float(configs['bbd_options']['positive_iou_threshold'])
        self.f_negIouLimit = float(configs['bbd_options']['negative_iou_limit'])
        self.i_topK = int(configs['bbd_options']['top_k'])
        self.i_nmsMaximumOutput = int(configs['bbd_options']['nms_maximum_output'])
        self.s_coordinatesType = configs['bbd_options']['coordinates_type']
        self.b_twoBoxesForAR1 = str2bool(configs['bbd_options']['two_boxes_for_AR1_switch'])
        self.b_clipBoxes = str2bool(configs['bbd_options']['clip_boxes_switch'])
        self.b_normalizeCoordinates = str2bool(configs['bbd_options']['normalize_coordinates_switch'])


class Callbacks(object):
    def __init__(self, saver, learning_rate, train_options):
        """
        Constructor class to prepare the callbacks for DLAE.
        :param configs: engine configuration structure
        """
        self.callbacks = []

        if saver.b_saveCkpt:
            checkpoints = keras.callbacks.ModelCheckpoint(saver.s_saveCkptPath,
                                                          period=saver.i_saveCkptFrequency, save_best_only=True)
            self.callbacks.append(checkpoints)

        if saver.b_saveCSV:
            csv_files = keras.callbacks.CSVLogger(saver.s_saveCSVPath)
            self.callbacks.append(csv_files)

        if saver.b_recordTensorboardLogs:
            tensorboard = keras.callbacks.TensorBoard(log_dir=saver.s_recordTensorboardPath,
                                                      histogram_freq=saver.i_recordTensorboardFrequency)
            self.callbacks.append(tensorboard)

        if train_options.b_earlyStop:
            early_stop = keras.callbacks.EarlyStopping(patience=train_options.i_earlyStopPatience)
            self.callbacks.append(early_stop)

        if learning_rate.b_lrDecayOnPlateau:
            reduce_on_plateau = keras.callbacks.ReduceLROnPlateau(patience=learning_rate.i_lrDecayOnPlateauPatience,
                                                                  factor=learning_rate.f_lrDecayOnPlateauFactor,
                                                                  min_lr=1e-9)
            self.callbacks.append(reduce_on_plateau)


class Augmentation(object):
    def __init__(self, configs):
        """
        Constructor class to prepare the training options for DLAE.
        :param configs: engine configuration structure
        """
        self.b_augmentation = literal_eval(configs['augmentation']['apply_augmentation_switch'])
        self.b_fw_centering = literal_eval(configs['augmentation']['featurewise_centering_switch'])
        self.b_sw_centering = literal_eval(configs['augmentation']['samplewise_centering_switch'])
        self.b_fw_normalization = literal_eval(configs['augmentation']['featurewise_normalization_switch'])
        self.b_sw_normalization = literal_eval(configs['augmentation']['samplewise_normalization_switch'])
        self.f_width_shift = float(configs['augmentation']['width_shift'])
        self.f_height_shift = float(configs['augmentation']['height_shift'])
        self.i_rotation_range = int(configs['augmentation']['rotation_range'])
        self.t_brightness_range = literal_eval(configs['augmentation']['brightness_range'])
        self.f_shear_range = float(configs['augmentation']['shear_range'])
        self.f_zoom_range = float(configs['augmentation']['zoom_range'])
        self.f_channel_shift_range = float(configs['augmentation']['channel_shift_range'])
        self.s_fill_mode = configs['augmentation']['fill_mode']
        self.f_cval = float(configs['augmentation']['cval'])
        self.b_horizontal_flip = literal_eval(configs['augmentation']['horizontal_flip_switch'])
        self.b_vertical_flip = literal_eval(configs['augmentation']['vertical_flip_switch'])
