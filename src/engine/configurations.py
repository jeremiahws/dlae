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


import keras
from ast import literal_eval
from src.utils.general_utils import str2bool, check_keys
from src.engine.loss_functions import TverskyLoss, SSDLoss, JaccardLoss, FocalLoss, SoftDiceLoss
from copy import deepcopy
from src.utils.data_generators import CNN2DDatasetGenerator, FCN2DDatasetGenerator,\
                                      SSD2DDatasetGenerator, CNN3DDatasetGenerator,\
                                      FCN3DDatasetGenerator


class EngineConfigurations(object):
    def __init__(self, configs):
        """
        Constructor class to build the engine configurations for DLAE.
        :param configs: engine configuration structure
        """
        self.dispatcher = Dispatcher(configs)
        self.data_preprocessing = Preprocessing(configs, self.dispatcher)
        self.augmentation = Augmentation(configs)
        self.train_options = TrainingOptions(configs)
        self.train_data = TrainData(configs, self.data_preprocessing, self.dispatcher, self.augmentation, self.train_options)
        self.val_data = ValidationData(configs, self.data_preprocessing, self.dispatcher, self.augmentation, self.train_options, self.train_data)
        self.test_data = TestData(configs, self.data_preprocessing, self.dispatcher, self.augmentation, self.train_options)
        self.learning_rate = LearningRate(configs)
        self.optimizer = Optimizer(configs, self.learning_rate)
        self.monitors = Monitors(configs)
        self.loader = Loader(configs)
        self.saver = Saver(configs)
        self.layers = Layers(configs)
        self.loss_function = LossFunction(configs, self.data_preprocessing)
        self.callbacks = Callbacks(self.saver, self.learning_rate, self.train_options)


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
        self.i_num_categories = int(configs['preprocessing']['categories'])
        self.b_weight_loss = str2bool(configs['preprocessing']['weight_loss_switch'])
        self.b_repeatX = str2bool(configs['preprocessing']['repeat_X_switch'])
        self.i_repeatX = int(configs['preprocessing']['repeat_X_quantity'])

        if dispatcher.type_signal == "Train" or dispatcher.type_signal == "Train from Checkpoint":
            self.b_prepare_train = True
            self.b_prepare_val = True
            self.b_prepare_test = False

        elif dispatcher.type_signal == "Inference":
            self.b_prepare_train = False
            self.b_prepare_val = False
            self.b_prepare_test = True


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
        self.i_random_seed = int(configs['augmentation']['random_seed'])
        self.i_rounds = int(configs['augmentation']['rounds'])
        self.f_zca_epsilon = literal_eval(configs['augmentation']['zca_epsilon'])


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


class TrainData(object):
    def __init__(self, configs, preprocessing, dispatcher, augmentation, train_optiions):
        """
        Constructor class to prepare the training data for DLAE.
        :param configs: engine configuration structure
        :param preprocessing: the preprocessing steps
        :param dispatcher: the engine dispatcher
        :param augmentation: the engine augmentation options
        :param train_optiions: the engine training options
        """
        self.errors = []
        self.warnings = []
        self.s_trainXPath = configs['paths']['train_X']
        self.s_trainYPath = configs['paths']['train_y']
        self.train_generator = None
        self.val_generator_reserve = None

        if (any(self.s_trainXPath) and any(self.s_trainYPath) is False)\
                or (any(self.s_trainYPath) and any(self.s_trainXPath) is False):
            self.errors.append('Level2Error:BothTrainImagesandAnnotationPathsNotSpecified')

        elif any(self.s_trainXPath) is False and any(self.s_trainYPath) is False:
            preprocessing.b_prepare_train = False

        else:
            proceed = check_keys(self.s_trainXPath, self.s_trainYPath)
            if proceed is False:
                self.errors.append('Level2Error:TrainXandTrainYDatasetNamesDontMatch')

        minimums = [0.]
        maximums = [0.]
        if preprocessing.s_normalization_type == 'global_x' or preprocessing.s_normalization_type == 'global_xy':
            if (any(preprocessing.f_minImageIntensity) is False) \
                    or (any(preprocessing.f_maxImageIntensity) is False) \
                    or (any(preprocessing.f_minImageIntensity) is False and any(
                        preprocessing.f_maxImageIntensity) is False):
                self.errors.append('Level2Error:SetBothMinandMaxImageIntensityforImageNormalization')
            else:
                mins = preprocessing.f_minImageIntensity.split(',')
                maxs = preprocessing.f_maxImageIntensity.split(',')
                minimums = []
                maximums = []
                for val in mins:
                    minimums.append(float(val))

                for val in maxs:
                    maximums.append(float(val))

        if preprocessing.b_prepare_train:
            if dispatcher.model_signal == 'CNN':
                # try:
                    if preprocessing.s_image_context == '2D':
                        generator = CNN2DDatasetGenerator(self.s_trainXPath,
                                                          self.s_trainYPath,
                                                          rotation_range=augmentation.i_rotation_range,
                                                          width_shift_range=augmentation.f_width_shift,
                                                          height_shift_range=augmentation.f_height_shift,
                                                          shear_range=augmentation.f_shear_range,
                                                          zoom_range=augmentation.f_zoom_range,
                                                          flip_horizontal=augmentation.b_horizontal_flip,
                                                          flip_vertical=augmentation.b_vertical_flip,
                                                          featurewise_center=augmentation.b_fw_centering,
                                                          featurewise_std_normalization=augmentation.b_fw_normalization,
                                                          samplewise_center=augmentation.b_sw_centering,
                                                          samplewise_std_normalization=augmentation.b_sw_normalization,
                                                          zca_epsilon=augmentation.f_zca_epsilon,
                                                          shuffle_data=train_optiions.b_shuffleData,
                                                          rounds=augmentation.i_rounds,
                                                          fill_mode=augmentation.s_fill_mode,
                                                          cval=augmentation.f_cval,
                                                          interpolation_order=1,
                                                          seed=augmentation.i_random_seed,
                                                          batch_size=train_optiions.i_batchSize,
                                                          validation_split=train_optiions.f_validationSplit,
                                                          subset='train',
                                                          normalization=preprocessing.s_normalization_type,
                                                          min_intensity=minimums,
                                                          max_intensity=maximums,
                                                          categorical_labels=preprocessing.b_to_categorical,
                                                          num_classes=preprocessing.i_num_categories,
                                                          repeat_chans=preprocessing.b_repeatX,
                                                          chan_repititions=preprocessing.i_repeatX)
                    elif preprocessing.s_image_context == '3D':
                        generator = CNN3DDatasetGenerator(self.s_trainXPath,
                                                          self.s_trainYPath,
                                                          rotation_range=augmentation.i_rotation_range,
                                                          width_shift_range=augmentation.f_width_shift,
                                                          height_shift_range=augmentation.f_height_shift,
                                                          shear_range=augmentation.f_shear_range,
                                                          zoom_range=augmentation.f_zoom_range,
                                                          flip_horizontal=augmentation.b_horizontal_flip,
                                                          flip_vertical=augmentation.b_vertical_flip,
                                                          featurewise_center=augmentation.b_fw_centering,
                                                          featurewise_std_normalization=augmentation.b_fw_normalization,
                                                          samplewise_center=augmentation.b_sw_centering,
                                                          samplewise_std_normalization=augmentation.b_sw_normalization,
                                                          zca_epsilon=augmentation.f_zca_epsilon,
                                                          shuffle_data=train_optiions.b_shuffleData,
                                                          rounds=augmentation.i_rounds,
                                                          fill_mode=augmentation.s_fill_mode,
                                                          cval=augmentation.f_cval,
                                                          interpolation_order=1,
                                                          seed=augmentation.i_random_seed,
                                                          batch_size=train_optiions.i_batchSize,
                                                          validation_split=train_optiions.f_validationSplit,
                                                          subset='train',
                                                          normalization=preprocessing.s_normalization_type,
                                                          min_intensity=minimums,
                                                          max_intensity=maximums,
                                                          categorical_labels=preprocessing.b_to_categorical,
                                                          num_classes=preprocessing.i_num_categories,
                                                          repeat_chans=preprocessing.b_repeatX,
                                                          chan_repititions=preprocessing.i_repeatX)
                    self.train_generator = deepcopy(generator)
                    if train_optiions.f_validationSplit > 0.:
                        self.val_generator_reserve = deepcopy(generator)
                        self.val_generator_reserve.subset = 'validation'
                # except:
                #     self.errors.append('Level2Error:CouldNotEstablishCNNTrainGenerator')
            elif dispatcher.model_signal == 'FCN' or dispatcher.model_signal == 'GAN':
                # try:
                    if preprocessing.s_image_context == '2D':
                        generator = FCN2DDatasetGenerator(self.s_trainXPath,
                                                          self.s_trainYPath,
                                                          rotation_range=augmentation.i_rotation_range,
                                                          width_shift_range=augmentation.f_width_shift,
                                                          height_shift_range=augmentation.f_height_shift,
                                                          shear_range=augmentation.f_shear_range,
                                                          zoom_range=augmentation.f_zoom_range,
                                                          flip_horizontal=augmentation.b_horizontal_flip,
                                                          flip_vertical=augmentation.b_vertical_flip,
                                                          featurewise_center=augmentation.b_fw_centering,
                                                          featurewise_std_normalization=augmentation.b_fw_normalization,
                                                          samplewise_center=augmentation.b_sw_centering,
                                                          samplewise_std_normalization=augmentation.b_sw_normalization,
                                                          zca_epsilon=augmentation.f_zca_epsilon,
                                                          shuffle_data=train_optiions.b_shuffleData,
                                                          rounds=augmentation.i_rounds,
                                                          fill_mode=augmentation.s_fill_mode,
                                                          cval=augmentation.f_cval,
                                                          interpolation_order=1,
                                                          seed=augmentation.i_random_seed,
                                                          batch_size=train_optiions.i_batchSize,
                                                          validation_split=train_optiions.f_validationSplit,
                                                          subset='train',
                                                          normalization=preprocessing.s_normalization_type,
                                                          min_intensity=minimums,
                                                          max_intensity=maximums,
                                                          categorical_labels=preprocessing.b_to_categorical,
                                                          num_classes=preprocessing.i_num_categories,
                                                          repeat_chans=preprocessing.b_repeatX,
                                                          chan_repititions=preprocessing.i_repeatX)
                    elif preprocessing.s_image_context == '3D':
                        generator = FCN3DDatasetGenerator(self.s_trainXPath,
                                                          self.s_trainYPath,
                                                          rotation_range=augmentation.i_rotation_range,
                                                          width_shift_range=augmentation.f_width_shift,
                                                          height_shift_range=augmentation.f_height_shift,
                                                          shear_range=augmentation.f_shear_range,
                                                          zoom_range=augmentation.f_zoom_range,
                                                          flip_horizontal=augmentation.b_horizontal_flip,
                                                          flip_vertical=augmentation.b_vertical_flip,
                                                          featurewise_center=augmentation.b_fw_centering,
                                                          featurewise_std_normalization=augmentation.b_fw_normalization,
                                                          samplewise_center=augmentation.b_sw_centering,
                                                          samplewise_std_normalization=augmentation.b_sw_normalization,
                                                          zca_epsilon=augmentation.f_zca_epsilon,
                                                          shuffle_data=train_optiions.b_shuffleData,
                                                          rounds=augmentation.i_rounds,
                                                          fill_mode=augmentation.s_fill_mode,
                                                          cval=augmentation.f_cval,
                                                          interpolation_order=1,
                                                          seed=augmentation.i_random_seed,
                                                          batch_size=train_optiions.i_batchSize,
                                                          validation_split=train_optiions.f_validationSplit,
                                                          subset='train',
                                                          normalization=preprocessing.s_normalization_type,
                                                          min_intensity=minimums,
                                                          max_intensity=maximums,
                                                          categorical_labels=preprocessing.b_to_categorical,
                                                          num_classes=preprocessing.i_num_categories,
                                                          repeat_chans=preprocessing.b_repeatX,
                                                          chan_repititions=preprocessing.i_repeatX)
                    self.train_generator = deepcopy(generator)
                    if train_optiions.f_validationSplit > 0.:
                        self.val_generator_reserve = deepcopy(generator)
                        self.val_generator_reserve.subset = 'validation'
                # except:
                #     self.errors.append('Level2Error:CouldNotEstablishFCNorGANTrainGenerator')

            elif dispatcher.model_signal == 'BBD':
                # try:
                    if preprocessing.s_image_context == '2D':
                        generator = SSD2DDatasetGenerator(self.s_trainXPath,
                                                          self.s_trainYPath,
                                                          shuffle_data=train_optiions.b_shuffleData,
                                                          rounds=augmentation.i_rounds,
                                                          seed=augmentation.i_random_seed,
                                                          batch_size=train_optiions.i_batchSize,
                                                          validation_split=train_optiions.f_validationSplit,
                                                          subset='train',
                                                          normalization=preprocessing.s_normalization_type,
                                                          min_intensity=minimums,
                                                          max_intensity=maximums,
                                                          repeat_chans=preprocessing.b_repeatX,
                                                          chan_repititions=preprocessing.i_repeatX)
                        self.train_generator = deepcopy(generator)
                        if train_optiions.f_validationSplit > 0.:
                            self.val_generator_reserve = deepcopy(generator)
                            self.val_generator_reserve.subset = 'validation'
                # except:
                #     self.errors.append('Level2Error:CouldNotEstablish2DBBDTrainGenerator')


class ValidationData(object):
    def __init__(self, configs, preprocessing, dispatcher, augmentation, train_optiions, train_data):
        """
        Constructor class to prepare the validation data for DLAE.
        :param configs: engine configuration structure
        :param preprocessing: the preprocessing steps
        :param dispatcher: the engine dispatcher
        :param augmentation: the engine augmentation options
        :param train_optiions: the engine training options
        :param train_data: the engine training data
        """
        self.errors = []
        self.warnings = []
        self.s_valXPath = configs['paths']['validation_X']
        self.s_valYPath = configs['paths']['validation_y']
        self.val_generator = None

        if train_data.val_generator_reserve is not None:
            self.val_generator = train_data.val_generator_reserve
        else:
            if (any(self.s_valXPath) and any(self.s_valYPath) is False) \
                    or (any(self.s_valYPath) and any(self.s_valXPath) is False):
                self.errors.append('Level2Error:BothValidationImagesandAnnotationPathsNotSpecified')

            elif any(self.s_valXPath) is False and any(self.s_valYPath) is False:
                preprocessing.b_prepare_val = False

            else:
                proceed = check_keys(self.s_valXPath, self.s_valYPath)
                if proceed is False:
                    self.errors.append('Level2Error:ValidationXandValidationYDatasetNamesDontMatch')

            minimums = [0.]
            maximums = [0.]
            if preprocessing.s_normalization_type == 'global_x' or preprocessing.s_normalization_type == 'global_xy':
                if (any(preprocessing.f_minImageIntensity) is False) \
                        or (any(preprocessing.f_maxImageIntensity) is False) \
                        or (any(preprocessing.f_minImageIntensity) is False and any(
                            preprocessing.f_maxImageIntensity) is False):
                    self.errors.append('Level2Error:SetBothMinandMaxImageIntensityforImageNormalization')
                else:
                    mins = preprocessing.f_minImageIntensity.split(',')
                    maxs = preprocessing.f_maxImageIntensity.split(',')
                    minimums = []
                    maximums = []
                    for val in mins:
                        minimums.append(float(val))

                    for val in maxs:
                        maximums.append(float(val))

            if preprocessing.b_prepare_val:
                if dispatcher.model_signal == 'CNN':
                    try:
                        if preprocessing.s_image_context == '2D':
                            generator = CNN2DDatasetGenerator(self.s_valXPath,
                                                              self.s_valYPath,
                                                              featurewise_center=augmentation.b_fw_centering,
                                                              featurewise_std_normalization=augmentation.b_fw_normalization,
                                                              samplewise_center=augmentation.b_sw_centering,
                                                              samplewise_std_normalization=augmentation.b_sw_normalization,
                                                              batch_size=train_optiions.i_batchSize,
                                                              subset='validation',
                                                              normalization=preprocessing.s_normalization_type,
                                                              min_intensity=minimums,
                                                              max_intensity=maximums,
                                                              categorical_labels=preprocessing.b_to_categorical,
                                                              num_classes=preprocessing.i_num_categories,
                                                              repeat_chans=preprocessing.b_repeatX,
                                                              chan_repititions=preprocessing.i_repeatX)
                        elif preprocessing.s_image_context == '3D':
                            generator = CNN3DDatasetGenerator(self.s_valXPath,
                                                              self.s_valYPath,
                                                              featurewise_center=augmentation.b_fw_centering,
                                                              featurewise_std_normalization=augmentation.b_fw_normalization,
                                                              samplewise_center=augmentation.b_sw_centering,
                                                              samplewise_std_normalization=augmentation.b_sw_normalization,
                                                              batch_size=train_optiions.i_batchSize,
                                                              subset='validation',
                                                              normalization=preprocessing.s_normalization_type,
                                                              min_intensity=minimums,
                                                              max_intensity=maximums,
                                                              categorical_labels=preprocessing.b_to_categorical,
                                                              num_classes=preprocessing.i_num_categories,
                                                              repeat_chans=preprocessing.b_repeatX,
                                                              chan_repititions=preprocessing.i_repeatX)
                        self.val_generator = deepcopy(generator)
                    except:
                        self.errors.append('Level2Error:CouldNotEstablishCnnValidationGenerator')
                elif dispatcher.model_signal == 'FCN' or dispatcher.model_signal == 'GAN':
                    try:
                        if preprocessing.s_image_context == '2D':
                            generator = FCN2DDatasetGenerator(self.s_valXPath,
                                                              self.s_valYPath,
                                                              featurewise_center=augmentation.b_fw_centering,
                                                              featurewise_std_normalization=augmentation.b_fw_normalization,
                                                              samplewise_center=augmentation.b_sw_centering,
                                                              samplewise_std_normalization=augmentation.b_sw_normalization,
                                                              batch_size=train_optiions.i_batchSize,
                                                              subset='validation',
                                                              normalization=preprocessing.s_normalization_type,
                                                              min_intensity=minimums,
                                                              max_intensity=maximums,
                                                              categorical_labels=preprocessing.b_to_categorical,
                                                              num_classes=preprocessing.i_num_categories,
                                                              repeat_chans=preprocessing.b_repeatX,
                                                              chan_repititions=preprocessing.i_repeatX)
                        elif preprocessing.s_image_context == '3D':
                            generator = FCN3DDatasetGenerator(self.s_valXPath,
                                                              self.s_valYPath,
                                                              featurewise_center=augmentation.b_fw_centering,
                                                              featurewise_std_normalization=augmentation.b_fw_normalization,
                                                              samplewise_center=augmentation.b_sw_centering,
                                                              samplewise_std_normalization=augmentation.b_sw_normalization,
                                                              batch_size=train_optiions.i_batchSize,
                                                              subset='validation',
                                                              normalization=preprocessing.s_normalization_type,
                                                              min_intensity=minimums,
                                                              max_intensity=maximums,
                                                              categorical_labels=preprocessing.b_to_categorical,
                                                              num_classes=preprocessing.i_num_categories,
                                                              repeat_chans=preprocessing.b_repeatX,
                                                              chan_repititions=preprocessing.i_repeatX)
                        self.val_generator = deepcopy(generator)
                    except:
                        self.errors.append('Level2Error:CouldNotEstablishFCNorGANValidationGenerator')
                elif dispatcher.model_signal == 'BBD':
                    try:
                        if preprocessing.s_image_context == '2D':
                            generator = SSD2DDatasetGenerator(self.s_valXPath,
                                                              self.s_valYPath,
                                                              batch_size=train_optiions.i_batchSize,
                                                              subset='validation',
                                                              normalization=preprocessing.s_normalization_type,
                                                              min_intensity=minimums,
                                                              max_intensity=maximums,
                                                              repeat_chans=preprocessing.b_repeatX,
                                                              chan_repititions=preprocessing.i_repeatX)
                            self.val_generator = deepcopy(generator)
                    except:
                        self.errors.append('Level2Error:CouldNotEstablish2DBBDValidationGenerator')


class TestData(object):
    def __init__(self, configs, preprocessing, dispatcher, augmentation, train_optiions):
        """
        Constructor class to prepare the testing data for DLAE.
        :param configs: engine configuration structure
        :param preprocessing: the preprocessing steps
        :param dispatcher: the engine dispatcher
        :param augmentation: the engine augmentation options
        :param train_optiions: the engine training options
        """
        self.errors = []
        self.warnings = []
        self.s_testXPath = configs['paths']['test_X']
        self.test_generator = None

        if any(self.s_testXPath) is False:
            pass
        else:
            minimums = [0.]
            maximums = [0.]
            if preprocessing.s_normalization_type == 'global_x' or preprocessing.s_normalization_type == 'global_xy':
                if (any(preprocessing.f_minImageIntensity) is False) \
                        or (any(preprocessing.f_maxImageIntensity) is False) \
                        or (any(preprocessing.f_minImageIntensity) is False and any(
                            preprocessing.f_maxImageIntensity) is False):
                    self.errors.append('Level2Error:SetBothMinandMaxImageIntensityforImageNormalization')
                else:
                    mins = preprocessing.f_minImageIntensity.split(',')
                    maxs = preprocessing.f_maxImageIntensity.split(',')
                    minimums = []
                    maximums = []
                    for val in mins:
                        minimums.append(float(val))

                    for val in maxs:
                        maximums.append(float(val))

            if preprocessing.b_prepare_test:
                if dispatcher.model_signal == 'CNN':
                    try:
                        if preprocessing.s_image_context == '2D':
                            generator = CNN2DDatasetGenerator(self.s_testXPath,
                                                              featurewise_center=augmentation.b_fw_centering,
                                                              featurewise_std_normalization=augmentation.b_fw_normalization,
                                                              samplewise_center=augmentation.b_sw_centering,
                                                              samplewise_std_normalization=augmentation.b_sw_normalization,
                                                              batch_size=train_optiions.i_batchSize,
                                                              subset='test',
                                                              normalization=preprocessing.s_normalization_type,
                                                              min_intensity=minimums[0],
                                                              max_intensity=maximums[0],
                                                              categorical_labels=preprocessing.b_to_categorical,
                                                              num_classes=preprocessing.i_num_categories,
                                                              repeat_chans=preprocessing.b_repeatX,
                                                              chan_repititions=preprocessing.i_repeatX)
                        elif preprocessing.s_image_context == '3D':
                            generator = CNN3DDatasetGenerator(self.s_testXPath,
                                                              featurewise_center=augmentation.b_fw_centering,
                                                              featurewise_std_normalization=augmentation.b_fw_normalization,
                                                              samplewise_center=augmentation.b_sw_centering,
                                                              samplewise_std_normalization=augmentation.b_sw_normalization,
                                                              batch_size=train_optiions.i_batchSize,
                                                              subset='test',
                                                              normalization=preprocessing.s_normalization_type,
                                                              min_intensity=minimums[0],
                                                              max_intensity=maximums[0],
                                                              categorical_labels=preprocessing.b_to_categorical,
                                                              num_classes=preprocessing.i_num_categories,
                                                              repeat_chans=preprocessing.b_repeatX,
                                                              chan_repititions=preprocessing.i_repeatX)
                        self.test_generator = deepcopy(generator)
                    except:
                        self.errors.append('Level2Error:CouldNotEstablishCNNTestGenerator')
                elif dispatcher.model_signal == 'FCN' or dispatcher.model_signal == 'GAN':
                    try:
                        if preprocessing.s_image_context == '2D':
                            generator = FCN2DDatasetGenerator(self.s_testXPath,
                                                              featurewise_center=augmentation.b_fw_centering,
                                                              featurewise_std_normalization=augmentation.b_fw_normalization,
                                                              samplewise_center=augmentation.b_sw_centering,
                                                              samplewise_std_normalization=augmentation.b_sw_normalization,
                                                              batch_size=train_optiions.i_batchSize,
                                                              subset='test',
                                                              normalization=preprocessing.s_normalization_type,
                                                              min_intensity=minimums[0],
                                                              max_intensity=maximums[0],
                                                              categorical_labels=preprocessing.b_to_categorical,
                                                              num_classes=preprocessing.i_num_categories,
                                                              repeat_chans=preprocessing.b_repeatX,
                                                              chan_repititions=preprocessing.i_repeatX)
                        if preprocessing.s_image_context == '3D':
                            generator = FCN3DDatasetGenerator(self.s_testXPath,
                                                              featurewise_center=augmentation.b_fw_centering,
                                                              featurewise_std_normalization=augmentation.b_fw_normalization,
                                                              samplewise_center=augmentation.b_sw_centering,
                                                              samplewise_std_normalization=augmentation.b_sw_normalization,
                                                              batch_size=train_optiions.i_batchSize,
                                                              subset='test',
                                                              normalization=preprocessing.s_normalization_type,
                                                              min_intensity=minimums[0],
                                                              max_intensity=maximums[0],
                                                              categorical_labels=preprocessing.b_to_categorical,
                                                              num_classes=preprocessing.i_num_categories,
                                                              repeat_chans=preprocessing.b_repeatX,
                                                              chan_repititions=preprocessing.i_repeatX)
                        self.test_generator = deepcopy(generator)
                    except:
                        self.errors.append('Level2Error:CouldNotEstablishFCNorGANTestGenerator')
                elif dispatcher.model_signal == 'BBD':
                    try:
                        if preprocessing.s_image_context == '2D':
                            generator = SSD2DDatasetGenerator(self.s_testXPath,
                                                              batch_size=train_optiions.i_batchSize,
                                                              subset='test',
                                                              normalization=preprocessing.s_normalization_type,
                                                              min_intensity=minimums,
                                                              max_intensity=maximums,
                                                              repeat_chans=preprocessing.b_repeatX,
                                                              chan_repititions=preprocessing.i_repeatX)
                            self.test_generator = deepcopy(generator)
                    except:
                        self.errors.append('Level2Error:CouldNotEstablish2DBBDTestGenerator')


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

        elif self.s_lossFunction == 'jaccard':
            jaccard_loss = JaccardLoss(smooth=self.f_parameter1)
            self.loss = jaccard_loss.compute_loss

        elif self.s_lossFunction == 'focal':
            focal_loss = FocalLoss(alpha=self.f_parameter1, gamma=self.f_parameter2)
            self.loss = focal_loss.compute_loss

        elif self.s_lossFunction == 'soft_dice':
            soft_dice_loss = SoftDiceLoss(smooth=self.f_parameter1)
            self.loss = soft_dice_loss.compute_loss


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
                                                                  cooldown=learning_rate.i_lrDecayOnPlateauPatience,
                                                                  min_lr=1e-9)
            self.callbacks.append(reduce_on_plateau)
