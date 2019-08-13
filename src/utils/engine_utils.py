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
"""src/utils/engine_utils.py"""


from src.engine.layers import *
import keras.backend.tensorflow_backend as K
from keras.layers import Concatenate
from src.engine.configurations import EngineConfigurations
from src.utils.general_utils import str2bool
import os
from ast import literal_eval


def l1_loss(y_true, y_pred):
    return K.sum(K.abs(y_pred - y_true))


def create_layer(layer_definition):
    layer_configs = layer_definition.split(':')
    layer_configs = ['None' if x is '' else x for x in layer_configs]

    if layer_configs[0] == 'Input':
        layer = InputLayer(layer_configs[1])

    elif layer_configs[0] == 'Reshape':
        layer = ReshapeLayer(layer_configs[1])

    elif layer_configs[0] == 'Dropout':
        layer = DropoutLayer(layer_configs[1])

    elif layer_configs[0] == 'Dense':
        layer = DenseLayer(layer_configs[1])

    elif layer_configs[0] == 'Activation':
        layer = ActivationLayer(layer_configs[1])

    elif layer_configs[0] == 'Permute':
        layer = PermuteLayer(layer_configs[1])

    elif layer_configs[0] == 'Flatten':
        layer = FlattenLayer()

    elif layer_configs[0] == 'Spatial dropout 2D':
        layer = SpatialDropout2DLayer(layer_configs[1])

    elif layer_configs[0] == 'Spatial dropout 3D':
        layer = SpatialDropout3DLayer(layer_configs[1])

    elif layer_configs[0] == 'Convolution 2D':
        layer = Conv2DLayer(layer_configs[1], layer_configs[2], layer_configs[3], layer_configs[4], layer_configs[5], layer_configs[6], layer_configs[7], layer_configs[8], layer_configs[9], layer_configs[10])

    elif layer_configs[0] == 'Separable convolution 2D':
        layer = SeparableConv2DLayer(layer_configs[1], layer_configs[2], layer_configs[3], layer_configs[4], layer_configs[5], layer_configs[6], layer_configs[7], layer_configs[8], layer_configs[9], layer_configs[10])

    elif layer_configs[0] == 'Depthwise separable convolution 2D':
        layer = DepthwiseSeparableConv2DLayer(layer_configs[1], layer_configs[2], layer_configs[3], layer_configs[4], layer_configs[5], layer_configs[6], layer_configs[7], layer_configs[8], layer_configs[9])

    elif layer_configs[0] == 'Transpose convolution 2D':
        layer = ConvTranspose2DLayer(layer_configs[1], layer_configs[2], layer_configs[3], layer_configs[4], layer_configs[5], layer_configs[6], layer_configs[7], layer_configs[8], layer_configs[9], layer_configs[10])

    elif layer_configs[0] == 'Resize convolution 2D':
        layer = ResizeConv2DLayer(layer_configs[1], layer_configs[2], layer_configs[3], layer_configs[4], layer_configs[5], layer_configs[6], layer_configs[7], layer_configs[8], layer_configs[9], layer_configs[10], layer_configs[11])

    elif layer_configs[0] == 'Convolution 3D':
        layer = Conv3DLayer(layer_configs[1], layer_configs[2], layer_configs[3], layer_configs[4], layer_configs[5], layer_configs[6], layer_configs[7], layer_configs[8], layer_configs[9], layer_configs[10])

    elif layer_configs[0] == 'Transpose convolution 3D':
        layer = ConvTranspose3DLayer(layer_configs[1], layer_configs[2], layer_configs[3], layer_configs[4], layer_configs[5], layer_configs[6], layer_configs[7], layer_configs[8], layer_configs[9], layer_configs[10])

    elif layer_configs[0] == 'Resize convolution 3D':
        layer = ResizeConv3DLayer(layer_configs[1], layer_configs[2], layer_configs[3], layer_configs[4], layer_configs[5], layer_configs[6], layer_configs[7], layer_configs[8], layer_configs[9], layer_configs[10], layer_configs[11])

    elif layer_configs[0] == 'Upsample 2D':
        layer = Upsample2DLayer(layer_configs[1])

    elif layer_configs[0] == 'Upsample 3D':
        layer = Upsample3DLayer(layer_configs[1])

    elif layer_configs[0] == 'Zero padding 2D':
        layer = ZeroPad2DLayer(layer_configs[1])

    elif layer_configs[0] == 'Zero padding 3D':
        layer = ZeroPad3DLayer(layer_configs[1])

    elif layer_configs[0] == 'Cropping 2D':
        layer = Cropping2DLayer(layer_configs[1])

    elif layer_configs[0] == 'Cropping 3D':
        layer = Cropping3DLayer(layer_configs[1])

    elif layer_configs[0] == 'Leaky reLU':
        layer = LeakyReluLayer(layer_configs[1])

    elif layer_configs[0] == 'ELU':
        layer = EluLayer(layer_configs[1])

    elif layer_configs[0] == 'Thresholded reLU':
        layer = ThresholdedReluLayer(layer_configs[1])

    elif layer_configs[0] == 'PreLU':
        layer = PreluLayer()

    elif layer_configs[0] == 'Max pooling 2D':
        layer = MaxPool2DLayer(layer_configs[1], layer_configs[2])

    elif layer_configs[0] == 'Average pooling 2D':
        layer = AvgPool2DLayer(layer_configs[1], layer_configs[2])

    elif layer_configs[0] == 'Global max pooling 2D':
        layer = GlobalMaxPool2DLayer()

    elif layer_configs[0] == 'Global average pooling 2D':
        layer = GlobalAvgPool2DLayer()

    elif layer_configs[0] == 'Max pooling 3D':
        layer = MaxPool3DLayer(layer_configs[1], layer_configs[2])

    elif layer_configs[0] == 'Average pooling 3D':
        layer = AvgPool3DLayer(layer_configs[1], layer_configs[2])

    elif layer_configs[0] == 'Global max pooling 3D':
        layer = GlobalMaxPool3DLayer()

    elif layer_configs[0] == 'Global average pooling 3D':
        layer = GlobalAvgPool3DLayer()

    elif layer_configs[0] == 'Batch normalization':
        layer = BatchNormalizationLayer(layer_configs[1], layer_configs[2])

    elif layer_configs[0] == 'Gaussian dropout':
        layer = GaussianDropoutLayer(layer_configs[1])

    elif layer_configs[0] == 'Gaussian noise':
        layer = GaussianNoiseLayer(layer_configs[1])

    elif layer_configs[0] == 'Alpha dropout':
        layer = AlphaDropoutLayer(layer_configs[1])

    elif layer_configs[0] == 'Outer skip source':
        layer = OuterSkipConnectionSourceLayer(layer_configs[1])

    elif layer_configs[0] == 'Outer skip target':
        layer = OuterSkipConnectionTargetLayer(layer_configs[1])

    elif layer_configs[0] == 'Inner skip source':
        layer = InnerSkipConnectionSourceLayer(layer_configs[1])

    elif layer_configs[0] == 'Inner skip target':
        layer = InnerSkipConnectionTargetLayer(layer_configs[1])

    elif layer_configs[0] == 'Hook connection source':
        layer = HookConnectionSourceLayer()

    elif layer_configs[0] == 'Xception':
        layer = XceptionLayer(layer_configs[1], layer_configs[2], layer_configs[3], layer_configs[4], layer_configs[5])

    elif layer_configs[0] == 'VGG16':
        layer = VGG16Layer(layer_configs[1], layer_configs[2], layer_configs[3], layer_configs[4], layer_configs[5])

    elif layer_configs[0] == 'VGG19':
        layer = VGG19Layer(layer_configs[1], layer_configs[2], layer_configs[3], layer_configs[4], layer_configs[5])

    elif layer_configs[0] == 'ResNet50':
        layer = ResNet50Layer(layer_configs[1], layer_configs[2], layer_configs[3], layer_configs[4], layer_configs[5])

    elif layer_configs[0] == 'ResNet101':
        layer = ResNet101Layer(layer_configs[1], layer_configs[2], layer_configs[3], layer_configs[4], layer_configs[5])

    elif layer_configs[0] == 'ResNet152':
        layer = ResNet152Layer(layer_configs[1], layer_configs[2], layer_configs[3], layer_configs[4], layer_configs[5])

    elif layer_configs[0] == 'ResNet50V2':
        layer = ResNet50V2Layer(layer_configs[1], layer_configs[2], layer_configs[3], layer_configs[4], layer_configs[5])

    elif layer_configs[0] == 'ResNet101V2':
        layer = ResNet101V2Layer(layer_configs[1], layer_configs[2], layer_configs[3], layer_configs[4], layer_configs[5])

    elif layer_configs[0] == 'ResNet152V2':
        layer = ResNet152V2Layer(layer_configs[1], layer_configs[2], layer_configs[3], layer_configs[4], layer_configs[5])

    elif layer_configs[0] == 'ResNeXt50':
        layer = ResNeXt50Layer(layer_configs[1], layer_configs[2], layer_configs[3], layer_configs[4], layer_configs[5])

    elif layer_configs[0] == 'ResNeXt101':
        layer = ResNeXt101Layer(layer_configs[1], layer_configs[2], layer_configs[3], layer_configs[4], layer_configs[5])

    elif layer_configs[0] == 'InceptionV3':
        layer = InceptionV3Layer(layer_configs[1], layer_configs[2], layer_configs[3], layer_configs[4], layer_configs[5])

    elif layer_configs[0] == 'InceptionResNetV2':
        layer = InceptionResNetV2Layer(layer_configs[1], layer_configs[2], layer_configs[3], layer_configs[4], layer_configs[5])

    elif layer_configs[0] == 'DenseNet121':
        layer = DenseNet121Layer(layer_configs[1], layer_configs[2], layer_configs[3], layer_configs[4], layer_configs[5])

    elif layer_configs[0] == 'DenseNet169':
        layer = DenseNet169Layer(layer_configs[1], layer_configs[2], layer_configs[3], layer_configs[4], layer_configs[5])

    elif layer_configs[0] == 'DenseNet201':
        layer = DenseNet201Layer(layer_configs[1], layer_configs[2], layer_configs[3], layer_configs[4], layer_configs[5])

    elif layer_configs[0] == 'MobileNet':
        layer = MobileNetLayer(layer_configs[1], layer_configs[2], layer_configs[3], layer_configs[4], layer_configs[5])

    elif layer_configs[0] == 'MobileNetV2':
        layer = MobileNetV2Layer(layer_configs[1], layer_configs[2], layer_configs[3], layer_configs[4], layer_configs[5])

    return layer


class ModelMGPU(keras.models.Model):
    '''
    Enable multi_gpu_model compatibility with ModelCheckpoiznt callback.
    :param ser_model: serial model
    :param gpus: number of GPUs

    Pulled from:
    https://github.com/keras-team/keras/issues/2436#issuecomment-354882296
    '''
    def __init__(self, ser_model, gpus):
        pmodel = keras.utils.multi_gpu_model(ser_model, gpus)
        self.__dict__.update(pmodel.__dict__)
        self._smodel = ser_model

    def __getattribute__(self, attrname):
        '''
        Override load and save methods to be used from the serial-model. The
        serial-model holds references to the weights in the multi-gpu model.
        '''
        if 'load' in attrname or 'save' in attrname:
            return getattr(self._smodel, attrname)

        return super(ModelMGPU, self).__getattribute__(attrname)


def level_one_error_checking(configs):
    errors = []
    warnings = []

    if any(configs['config_file']['model_signal'] in x for x in ['CNN', 'FCN', 'GAN', 'BBD']):
        pass
    else:
        errors.append('Level1Error:NonexistentDispatcherModelSignal')

    if any(configs['config_file']['type_signal'] in x for x in ['Train', 'Train from Checkpoint', 'Inference']):
        pass
    else:
        errors.append('Level1Error:NonexistentDispatcherTypeSignal')

    if any(configs['config_file']['input_shape']):
        try:
            if type(literal_eval(configs['config_file']['input_shape'])) is not tuple:
                errors.append('Level1Error:InputShapeShouldBeTuple')
        except ValueError:
            errors.append('Level1Error:InputShapeShouldBeTuple')
    else:
        errors.append('Level1Error:MustDefineInputShape')

    if os.path.exists(configs['paths']['load_config']) or any(configs['paths']['load_config']) is False:
        pass
    else:
        errors.append('Level1Error:LoadConfigurationPathDoesNotExist')

    if os.path.exists(configs['paths']['load_checkpoint']) or any(configs['paths']['load_checkpoint']) is False:
        pass
    else:
        errors.append('Level1Error:LoadCheckpointPathDoesNotExist')

    if os.path.exists(configs['paths']['load_model']) or any(configs['paths']['load_model']) is False:
        pass
    else:
        errors.append('Level1Error:LoadModelPathDoesNotExist')

    if os.path.exists(configs['paths']['train_X']) or any(configs['paths']['train_X']) is False:
        pass
    else:
        errors.append('Level1Error:LoadTrainXPathDoesNotExist')

    if os.path.exists(configs['paths']['train_y']) or any(configs['paths']['train_y']) is False:
        pass
    else:
        errors.append('Level1Error:LoadTrainyPathDoesNotExist')

    if os.path.exists(configs['paths']['validation_X']) or any(configs['paths']['validation_X']) is False:
        pass
    else:
        errors.append('Level1Error:LoadValidationXPathDoesNotExist')

    if os.path.exists(configs['paths']['validation_y']) or any(configs['paths']['validation_y']) is False:
        pass
    else:
        errors.append('Level1Error:LoadValidationyPathDoesNotExist')

    if os.path.exists(configs['paths']['test_X']) or any(configs['paths']['test_X']) is False:
        pass
    else:
        errors.append('Level1Error:LoadTestXPathDoesNotExist')

    if any(configs['preprocessing']['image_context'] in x for x in ['2D', '3D']):
        pass
    else:
        errors.append('Level1Error:NonexistentImageContext')

    if any(configs['preprocessing']['normalization_type'] in x for x in ['X from [0, 1]', 'X from [-1, 1]', 'X, Y from [0, 1]', 'X, Y from [-1, 1]', 'none']):
        if any(configs['preprocessing']['normalization_type'] in x for x in ['X from [0, 1]', 'X from [-1, 1]']):
            if any(configs['preprocessing']['minimum_image_intensity']):
                try:
                    float(configs['preprocessing']['minimum_image_intensity'])
                except ValueError:
                    errors.append('Warning:MinimumImageIntensityShouldBeFloat')
            else:
                errors.append('Level1Error:SpecifyMinimumImageIntensitytoPerformNormalization')

            if any(configs['preprocessing']['maximum_image_intensity']):
                try:
                    float(configs['preprocessing']['maximum_image_intensity'])
                except ValueError:
                    errors.append('Warning:MaximumImageIntensityShouldBeFloat')
            else:
                errors.append('Level1Error:SpecifyMaximumImageIntensitytoPerformNormalization')

        elif any(configs['preprocessing']['normalization_type'] in x for x in ['X, Y from [0, 1]', 'X, Y from [-1, 1]']):
            try:
                minimums = configs['preprocessing']['minimum_image_intensity'].split(',')
                if len(minimums) == 2:
                    try:
                        float(minimums[0])
                        float(minimums[1])
                    except ValueError:
                        errors.append('Level1Error:SpecifyTwoValuesSeparatedbyCommaforMinimumImageIntensityforSelectedImageNormalization')
                else:
                    errors.append('Level1Error:SpecifyTwoValuesSeparatedbyCommaforMinimumImageIntensityforSelectedImageNormalization')
            except ValueError:
                errors.append('Level1Error:SpecifyTwoValuesSeparatedbyCommaforMinimumImageIntensityforSelectedImageNormalization')

            try:
                maximums = configs['preprocessing']['maximum_image_intensity'].split(',')
                if len(maximums) == 2:
                    try:
                        float(maximums[0])
                        float(maximums[1])
                    except ValueError:
                        errors.append('Level1Error:SpecifyTwoValuesSeparatedbyCommaforMaximumImageIntensityforSelectedImageNormalization')
                else:
                    errors.append('Level1Error:SpecifyTwoValuesSeparatedbyCommaforMaximumImageIntensityforSelectedImageNormalization')
            except ValueError:
                errors.append('Level1Error:SpecifyTwoValuesSeparatedbyCommaforMaximumImageIntensityforSelectedImageNormalization')
    else:
        errors.append('Level1Error:NonexistentImageNormalizationType')

    try:
        str2bool(configs['preprocessing']['categorical_switch'])
    except ValueError:
        warnings.append('Warning:ConvertToCategoricalSwitchShouldBeBool')
        configs['preprocessing']['categorical_switch'] = 'False'

    try:
        str2bool(configs['preprocessing']['weight_loss_switch'])
    except ValueError:
        warnings.append('Warning:WeightLossFunctionSwitchShouldBeBool')
        configs['preprocessing']['weight_loss_switch'] = 'False'

    try:
        str2bool(configs['preprocessing']['reshape_X_switch'])
    except ValueError:
        warnings.append('Warning:ReshapeXSwitchShouldBeBool')
        configs['preprocessing']['reshape_X_switch'] = 'False'

    if any(configs['preprocessing']['reshape_X_dimensions']):
        try:
            if type(literal_eval(configs['preprocessing']['reshape_X_dimensions'])) is not tuple:
                errors.append('Level1Error:ReshapeXDimensionsShouldBeTuple')
        except ValueError:
            errors.append('Level1Error:ReshapeXDimensionsShouldBeTuple')

    try:
        str2bool(configs['preprocessing']['permute_X_switch'])
    except ValueError:
        warnings.append('Warning:PermuteXSwitchShouldBeBool')
        configs['preprocessing']['permute_X_switch'] = 'False'

    if any(configs['preprocessing']['permute_X_dimensions']):
        try:
            if type(literal_eval(configs['preprocessing']['permute_X_dimensions'])) is not tuple:
                errors.append('Level1Error:PermuteXDimensionsShouldBeTuple')
        except ValueError:
            errors.append('Level1Error:PermuteXDimensionsShouldBeTuple')

    try:
        str2bool(configs['preprocessing']['repeat_X_switch'])
    except ValueError:
        warnings.append('Warning:RepeatXSwitchShouldBeBool')
        configs['preprocessing']['repeat_X_switch'] = 'False'

    if any(configs['preprocessing']['repeat_X_quantity']):
        try:
            int(configs['preprocessing']['repeat_X_quantity'])
        except ValueError:
            warnings.append('Warning:RepeatXQuantityShouldBeInt')

    try:
        str2bool(configs['preprocessing']['reshape_y_switch'])
    except ValueError:
        warnings.append('Warning:ReshapeySwitchShouldBeBool')
        configs['preprocessing']['reshape_y_switch'] = 'False'

    if any(configs['preprocessing']['reshape_y_dimensions']):
        try:
            if type(literal_eval(configs['preprocessing']['reshape_y_dimensions'])) is not tuple:
                errors.append('Level1Error:ReshapeyDimensionsShouldBeTuple')
        except ValueError:
            errors.append('Level1Error:ReshapeyDimensionsShouldBeTuple')

    try:
        str2bool(configs['preprocessing']['permute_y_switch'])
    except ValueError:
        warnings.append('Warning:PermuteySwitchShouldBeBool')
        configs['preprocessing']['permute_y_switch'] = 'False'

    if any(configs['preprocessing']['permute_y_dimensions']):
        try:
            if type(literal_eval(configs['preprocessing']['permute_y_dimensions'])) is not tuple:
                errors.append('Level1Error:PermuteyDimensionsShouldBeTuple')
        except ValueError:
            errors.append('Level1Error:PermuteyDimensionsShouldBeTuple')

    try:
        str2bool(configs['augmentation']['apply_augmentation_switch'])
    except ValueError:
        warnings.append('Warning:ApplyAugmentationSwitchShouldBeBool')
        configs['augmentation']['apply_augmentation_switch'] = 'False'

    try:
        str2bool(configs['augmentation']['featurewise_centering_switch'])
    except ValueError:
        warnings.append('Warning:FeaturewiseCenteringSwitchShouldBeBool')
        configs['augmentation']['featurewise_centering_switch'] = 'False'

    try:
        str2bool(configs['augmentation']['samplewise_centering_switch'])
    except ValueError:
        warnings.append('Warning:SamplewiseCenteringSwitchShouldBeBool')
        configs['augmentation']['samplewise_centering_switch'] = 'False'

    try:
        str2bool(configs['augmentation']['featurewise_normalization_switch'])
    except ValueError:
        warnings.append('Warning:FeaturewiseNormalizationSwitchShouldBeBool')
        configs['augmentation']['featurewise_normalization_switch'] = 'False'

    try:
        str2bool(configs['augmentation']['samplewise_normalization_switch'])
    except ValueError:
        warnings.append('Warning:SamplewiseNormalizationSwitchShouldBeBool')
        configs['augmentation']['samplewise_normalization_switch'] = 'False'

    if any(configs['augmentation']['width_shift']):
        try:
            float(configs['augmentation']['width_shift'])
        except ValueError:
            warnings.append('Warning:WidthShiftShouldBeFloat')
    else:
        configs['augmentation']['width_shift'] = 0.1

    if any(configs['augmentation']['height_shift']):
        try:
            float(configs['augmentation']['height_shift'])
        except ValueError:
            warnings.append('Warning:HeightShiftShouldBeFloat')
    else:
        configs['augmentation']['height_shift'] = 0.1

    if any(configs['augmentation']['rotation_range']):
        try:
            int(configs['augmentation']['rotation_range'])
        except ValueError:
            warnings.append('Warning:RotationRangeShouldBeInt')
    else:
        configs['augmentation']['rotation_range'] = 0

    if any(configs['augmentation']['brightness_range']):
        try:
            if type(literal_eval(configs['augmentation']['brightness_range'])) is tuple\
                    or literal_eval(configs['augmentation']['brightness_range']) is None:
                pass
        except ValueError:
            warnings.append('Warning:OptimizerBrightnessRangeShouldBeTupleorNone')
            configs['augmentation']['brightness_range'] = 'None'
    else:
        configs['augmentation']['brightness_range'] = 'None'

    if any(configs['augmentation']['shear_range']):
        try:
            float(configs['augmentation']['shear_range'])
        except ValueError:
            warnings.append('Warning:ShearRangeShouldBeFloat')
    else:
        configs['augmentation']['shear_range'] = 0.0

    if any(configs['augmentation']['zoom_range']):
        try:
            float(configs['augmentation']['zoom_range'])
        except ValueError:
            warnings.append('Warning:ZoomRangeShouldBeFloat')
    else:
        configs['augmentation']['zoom_range'] = 0.0

    if any(configs['augmentation']['channel_shift_range']):
        try:
            float(configs['augmentation']['channel_shift_range'])
        except ValueError:
            warnings.append('Warning:ChannelShiftRangeShouldBeFloat')
    else:
        configs['augmentation']['channel_shift_range'] = 0.0

    if any(configs['augmentation']['fill_mode'] in x for x in ['nearest', 'constant', 'reflect', 'wrap']):
        pass
    else:
        errors.append('Level1Error:NonexistentFillMode')

    if any(configs['augmentation']['cval']):
        try:
            float(configs['augmentation']['cval'])
        except ValueError:
            warnings.append('Warning:CvalShouldBeFloat')
    else:
        configs['augmentation']['cval'] = 0.0

    try:
        str2bool(configs['augmentation']['horizontal_flip_switch'])
    except ValueError:
        warnings.append('Warning:HorizontalFlipSwitchShouldBeBool')
        configs['augmentation']['horizontal_flip_switch'] = 'False'

    try:
        str2bool(configs['augmentation']['vertical_flip_switch'])
    except ValueError:
        warnings.append('Warning:VerticalFlipSwitchShouldBeBool')
        configs['augmentation']['vertical_flip_switch'] = 'False'

    if any(configs['loss_function']['loss'] in x for x in ['categorical_crossentropy',
                                                           'sparse_categorical_crossentropy', 'mean_squared_error',
                                                           'mean_absolute_error', 'tversky', 'pix2pix',
                                                           'cyclegan', 'ssd']):
        pass
    else:
        errors.append('Level1Error:NonexistentLossFunction')

    if any(configs['loss_function']['parameter1']):
        try:
            float(configs['loss_function']['parameter1'])
        except ValueError:
            warnings.append('Warning:Parameter1ShouldBeFloat')
    else:
        configs['loss_function']['parameter1'] = 0.0

    if any(configs['loss_function']['parameter2']):
        try:
            float(configs['loss_function']['parameter2'])
        except ValueError:
            warnings.append('Warning:Parameter2ShouldBeFloat')
    else:
        configs['loss_function']['parameter2'] = 0.0

    if any(configs['learning_rate_schedule']['learning_rate']):
        try:
            float(configs['learning_rate_schedule']['learning_rate'])
        except ValueError:
            warnings.append('Warning:LearningRateShouldBeFloat')
    else:
        configs['learning_rate_schedule']['learning_rate'] = 0.0001

    if any(configs['learning_rate_schedule']['learning_rate_decay_factor']):
        try:
            float(configs['learning_rate_schedule']['learning_rate_decay_factor'])
        except ValueError:
            warnings.append('Warning:LearningRateDecayFactorShouldBeFloat')
    else:
        configs['learning_rate_schedule']['learning_rate_decay_factor'] = 0.0

    try:
        str2bool(configs['learning_rate_schedule']['decay_on_plateau_switch'])
    except ValueError:
        warnings.append('Warning:DecayOnPlateauSwitchShouldBeBool')
        configs['learning_rate_schedule']['decay_on_plateau_switch'] = 'False'

    if any(configs['learning_rate_schedule']['decay_on_plateau_factor']):
        try:
            float(configs['learning_rate_schedule']['decay_on_plateau_factor'])
        except ValueError:
            warnings.append('Warning:DecayOnPlateauFactorShouldBeFloat')
    else:
        configs['learning_rate_schedule']['decay_on_plateau_factor'] = 0.0

    if any(configs['learning_rate_schedule']['decay_on_plateau_patience']):
        try:
            int(configs['learning_rate_schedule']['decay_on_plateau_patience'])
        except ValueError:
            warnings.append('Warning:DecayOnPlateauPatienceShouldBeInt')
    else:
        configs['learning_rate_schedule']['decay_on_plateau_patience'] = 3

    try:
        str2bool(configs['learning_rate_schedule']['step_decay_switch'])
    except ValueError:
        warnings.append('Warning:StepDecaySwitchShouldBeBool')
        configs['learning_rate_schedule']['step_decay_switch'] = 'False'

    if any(configs['learning_rate_schedule']['step_decay_factor']):
        try:
            float(configs['learning_rate_schedule']['step_decay_factor'])
        except ValueError:
            warnings.append('Warning:StepDecayFactorShouldBeFloat')
    else:
        configs['learning_rate_schedule']['step_decay_factor'] = 0.0

    if any(configs['learning_rate_schedule']['step_decay_period']):
        try:
            int(configs['learning_rate_schedule']['step_decay_period'])
        except ValueError:
            warnings.append('Warning:StepDecayPeriodShouldBeInt')
    else:
        configs['learning_rate_schedule']['step_decay_period'] = 3

    if any(configs['learning_rate_schedule']['discriminator_learning_rate']):
        try:
            values = configs['learning_rate_schedule']['discriminator_learning_rate'].split(':')
            if type(literal_eval(values[0])) is float:
                pass
            else:
                warnings.append('Warning:DiscriminatorLearningRateShouldBeFloat')
                values[0] = '0.0001'
            if type(literal_eval(values[1])) is float:
                pass
            else:
                warnings.append('Warning:DiscriminatorLearningRateDecayShouldBeFloat')
                values[1] = '0.0'
            configs['learning_rate_schedule']['discriminator_learning_rate'] = ':'.join([values[0], values[1]])
        except ValueError:
            errors.append('Level1Error:CannotDetermineDiscriminatorLearningRateConfigurations')
    else:
        configs['learning_rate_schedule']['discriminator_learning_rate'] = '0.0001:0.0'

    if any(configs['learning_rate_schedule']['gan_learning_rate']):
        try:
            values = configs['learning_rate_schedule']['gan_learning_rate'].split(':')
            if type(literal_eval(values[0])) is float:
                pass
            else:
                warnings.append('Warning:GANLearningRateShouldBeFloat')
                values[0] = '0.0001'
            if type(literal_eval(values[0])) is float:
                pass
            else:
                warnings.append('Warning:GANLearningRateDecayShouldBeFloat')
                values[1] = '0.0'
            configs['learning_rate_schedule']['gan_learning_rate'] = ':'.join([values[0], values[1]])
        except ValueError:
            errors.append('Level1Error:CannotDetermineGANLearningRateConfigurations')
    else:
        configs['learning_rate_schedule']['gan_learning_rate'] = '0.0001:0.0'

    if any(configs['optimizer']['optimizer'] in x for x in ['Adam', 'NAdam', 'SGD', 'RMSprop',
                                                            'Adagrad', 'Adadelta', 'Adamax']):
        pass
    else:
        errors.append('Level1Error:NonexistentOptimizer')

    if any(configs['optimizer']['beta1']):
        try:
            float(configs['optimizer']['beta1'])
        except ValueError:
            warnings.append('Warning:OptimizerBeta1ShouldBeFloat')
            configs['optimizer']['beta1'] = '0.9'
    else:
        configs['optimizer']['beta1'] = '0.9'

    if any(configs['optimizer']['beta2']):
        try:
            float(configs['optimizer']['beta2'])
        except ValueError:
            warnings.append('Warning:OptimizerBeta2ShouldBeFloat')
            configs['optimizer']['beta2'] = '0.999'
    else:
        configs['optimizer']['beta2'] = '0.999'

    if any(configs['optimizer']['rho']):
        try:
            float(configs['optimizer']['rho'])
        except ValueError:
            warnings.append('Warning:OptimizerRhoShouldBeFloat')
            configs['optimizer']['rho'] = '0.9'
    else:
        configs['optimizer']['rho'] = '0.9'

    if any(configs['optimizer']['momentum']):
        try:
            float(configs['optimizer']['momentum'])
        except ValueError:
            warnings.append('Warning:OptimizerMomentumShouldBeFloat')
            configs['optimizer']['momentum'] = '0.0'
    else:
        configs['optimizer']['momentum'] = '0.0'

    if any(configs['optimizer']['epsilon']):
        try:
            if type(literal_eval(configs['optimizer']['epsilon'])) is float\
                    or literal_eval(configs['optimizer']['epsilon']) is None:
                pass
        except ValueError:
            warnings.append('Warning:OptimizerEpsilonShouldBeFloatorNone')
            configs['optimizer']['epsilon'] = 'None'
    else:
        configs['optimizer']['epsilon'] = 'None'

    if any(configs['optimizer']['discriminator_optimizer']):
        try:
            values = configs['optimizer']['discriminator_optimizer'].split(':')
            if any(values[0] in x for x in ['Adam', 'NAdam', 'SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adamax']):
                pass
            else:
                errors.append('Level1Error:NonexistentDiscriminatorOptimizer')
            if type(literal_eval(values[1])) is float:
                pass
            else:
                warnings.append('Warning:DiscriminatorOptimizerBeta1ShouldBeFloat')
                values[1] = '0.9'
            if type(literal_eval(values[2])) is float:
                pass
            else:
                warnings.append('Warning:DiscriminatorOptimizerBeta2ShouldBeFloat')
                values[2] = '0.999'
            if type(literal_eval(values[3])) is float:
                pass
            else:
                warnings.append('Warning:DiscriminatorOptimizerRhoShouldBeFloat')
                values[3] = '0.9'
            if type(literal_eval(values[4])) is float:
                pass
            else:
                warnings.append('Warning:DiscriminatorOptimizerMomentumShouldBeFloat')
                values[4] = '0.0'
            if type(literal_eval(values[5])) is float or literal_eval(values[5]) is None:
                pass
            else:
                warnings.append('Warning:DiscriminatorOptimizerEpsilonShouldBeFloatorNone')
                values[5] = 'None'
            configs['optimizer']['discriminator_optimizer'] = ':'.join([values[0], values[1], values[2],
                                                                        values[3], values[4], values[5]])
        except ValueError:
            errors.append('Level1Error:CannotDetermineDiscriminatorOptimizerConfigurations')
    else:
        configs['optimizer']['discriminator_optimizer'] = 'Adam:0.9:0.999:0.9:0.0:None'

    if any(configs['optimizer']['gan_optimizer']):
        try:
            values = configs['optimizer']['gan_optimizer'].split(':')
            if any(values[0] in x for x in ['Adam', 'NAdam', 'SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adamax']):
                pass
            else:
                errors.append('Level1Error:NonexistentGANOptimizer')
            if type(literal_eval(values[1])) is float:
                pass
            else:
                warnings.append('Warning:GANOptimizerBeta1ShouldBeFloat')
                values[1] = '0.9'
            if type(literal_eval(values[2])) is float:
                pass
            else:
                warnings.append('Warning:GANOptimizerBeta2ShouldBeFloat')
                values[2] = '0.999'
            if type(literal_eval(values[3])) is float:
                pass
            else:
                warnings.append('Warning:GANOptimizerRhoShouldBeFloat')
                values[3] = '0.9'
            if type(literal_eval(values[4])) is float:
                pass
            else:
                warnings.append('Warning:GANOptimizerMomentumShouldBeFloat')
                values[4] = '0.0'
            if type(literal_eval(values[5])) is float or literal_eval(values[5]) is None:
                pass
            else:
                warnings.append('Warning:GANOptimizerEpsilonShouldBeFloatorNone')
                values[5] = 'None'
            configs['optimizer']['gan_optimizer'] = ':'.join([values[0], values[1], values[2],
                                                              values[3], values[4], values[5]])
        except ValueError:
            errors.append('Level1Error:CannotDetermineGANOptimizerConfigurations')
    else:
        configs['optimizer']['gan_optimizer'] = 'Adam:0.9:0.999:0.9:0.0:None'

    if any(configs['training_configurations']['hardware'] in x for x in ['gpu', 'multi-gpu', 'cpu']):
        pass
    else:
        errors.append('Level1Error:NonexistentHardware')

    if any(configs['training_configurations']['number_of_gpus']):
        try:
            int(configs['training_configurations']['number_of_gpus'])
        except ValueError:
            warnings.append('Warning:NumberOfGpusShouldBeInt')
            configs['training_configurations']['number_of_gpus'] = '1'
    else:
        configs['training_configurations']['number_of_gpus'] = '1'

    try:
        str2bool(configs['training_configurations']['early_stop_switch'])
    except ValueError:
        warnings.append('Warning:EarlyStopSwitchShouldBeBool')
        configs['training_configurations']['early_stop_switch'] = 'False'

    if any(configs['training_configurations']['early_stop_patience']):
        try:
            int(configs['training_configurations']['early_stop_patience'])
        except ValueError:
            warnings.append('Warning:EarlyStopPatienceShouldBeInt')
            configs['training_configurations']['early_stop_patience'] = '10'
    else:
        configs['training_configurations']['early_stop_patience'] = '10'

    if any(configs['training_configurations']['batch_size']):
        try:
            int(configs['training_configurations']['batch_size'])
        except ValueError:
            warnings.append('Warning:BatchSizeShouldBeInt')
            configs['training_configurations']['batch_size'] = '32'
    else:
        configs['training_configurations']['batch_size'] = '32'

    if any(configs['training_configurations']['epochs']):
        try:
            int(configs['training_configurations']['epochs'])
        except ValueError:
            warnings.append('Warning:EpochsShouldBeInt')
            configs['training_configurations']['epochs'] = '500'
    else:
        configs['training_configurations']['epochs'] = '500'

    try:
        str2bool(configs['training_configurations']['shuffle_data_switch'])
    except ValueError:
        warnings.append('Warning:ShuffleDataSwitchShouldBeBool')
        configs['training_configurations']['shuffle_data_switch'] = 'True'

    if any(configs['training_configurations']['validation_split']):
        try:
            float(configs['training_configurations']['validation_split'])
        except ValueError:
            warnings.append('Warning:ValidationSplitShouldBeFloat')
            configs['training_configurations']['validation_split'] = '0.0'
    else:
        configs['training_configurations']['validation_split'] = '0.0'

    try:
        str2bool(configs['monitors']['mse_switch'])
    except ValueError:
        warnings.append('Warning:MSESwitchShouldBeBool')
        configs['monitors']['mse_switch'] = 'False'

    try:
        str2bool(configs['monitors']['mae_switch'])
    except ValueError:
        warnings.append('Warning:MAESwitchShouldBeBool')
        configs['monitors']['mae_switch'] = 'False'

    try:
        str2bool(configs['monitors']['accuracy_switch'])
    except ValueError:
        warnings.append('Warning:AccuracySwitchShouldBeBool')
        configs['monitors']['accuracy_switch'] = 'True'

    try:
        str2bool(configs['save_configurations']['save_model_switch'])
    except ValueError:
        warnings.append('Warning:SaveModelSwitchShouldBeBool')
        configs['save_configurations']['save_model_switch'] = 'False'

    if any(configs['save_configurations']['save_model_path']):
        if os.path.exists(os.path.dirname(configs['save_configurations']['save_model_path'])) is False:
            errors.append('Level1Error:NonexistentSaveModelDirectory')

        file, ext = os.path.splitext(configs['save_configurations']['save_model_path'])
        if ext != '.h5':
            warnings.append('Warning:SaveModelFileExtensionMustBeh5')
            configs['save_configurations']['save_model_path'] = file + '.h5'

    try:
        str2bool(configs['save_configurations']['save_csv_switch'])
    except ValueError:
        warnings.append('Warning:SaveCSVSwitchShouldBeBool')
        configs['save_configurations']['save_csv_switch'] = 'False'

    if any(configs['save_configurations']['save_csv_path']):
        if os.path.exists(os.path.dirname(configs['save_configurations']['save_csv_path'])) is False:
            errors.append('Level1Error:NonexistentSaveCSVDirectory')

        file, ext = os.path.splitext(configs['save_configurations']['save_csv_path'])
        if ext != '.csv':
            warnings.append('Warning:SaveCSVFileExtensionMustBecsv')
            configs['save_configurations']['save_csv_path'] = file + '.csv'

    try:
        str2bool(configs['save_configurations']['save_checkpoints_switch'])
    except ValueError:
        warnings.append('Warning:SaveModelCheckpointsSwitchShouldBeBool')
        configs['save_configurations']['save_checkpoints_switch'] = 'False'

    if any(configs['save_configurations']['save_checkpoints_path']):
        if os.path.exists(os.path.dirname(configs['save_configurations']['save_checkpoints_path'])) is False:
            errors.append('Level1Error:NonexistentSaveModelCheckpointsDirectory')

        file, ext = os.path.splitext(configs['save_configurations']['save_checkpoints_path'])
        if ext != '.h5':
            warnings.append('Warning:SaveModelCheckpointsFileExtensionMustBeh5')
            configs['save_configurations']['save_checkpoints_path'] = file + '.h5'

    if any(configs['save_configurations']['save_checkpoints_frequency']):
        try:
            int(configs['save_configurations']['save_checkpoints_frequency'])
        except ValueError:
            warnings.append('Warning:SaveCheckpointsFrequencyShouldBeInt')

    try:
        str2bool(configs['save_configurations']['save_tensorboard_switch'])
    except ValueError:
        warnings.append('Warning:SaveTensorboardSwitchShouldBeBool')
        configs['save_configurations']['save_tensorboard_switch'] = 'False'

    if any(configs['save_configurations']['save_tensorboard_path']):
        if os.path.exists(os.path.dirname(configs['save_configurations']['save_tensorboard_path'])) is False:
            errors.append('Level1Error:NonexistentSaveTensorboardDirectory')

    if any(configs['save_configurations']['save_tensorboard_frequency']):
        try:
            int(configs['save_configurations']['save_tensorboard_frequency'])
        except ValueError:
            warnings.append('Warning:SaveTensorboardFrequencyShouldBeInt')

    if any(configs['layers']['serial_layer_list']):
        for layer in configs['layers']['serial_layer_list']:
            if type(layer) is not str:
                errors.append('Level1Error:SerialLayersListContainsInvalidLayer')
                break

    if any(configs['layers']['generator_layer_list']):
        for layer in configs['layers']['generator_layer_list']:
            if type(layer) is not str:
                errors.append('Level1Error:GeneratorLayersListContainsInvalidLayer')
                break

    if any(configs['layers']['discriminator_layer_list']):
        for layer in configs['layers']['discriminator_layer_list']:
            if type(layer) is not str:
                errors.append('Level1Error:DiscriminatorLayersListContainsInvalidLayer')
                break

    if any(configs['bbd_options']['scaling_type'] in x for x in ['global', 'per predictor layer']):
        pass
    else:
        errors.append('Level1Error:NonexistentScalingType')

    if any(configs['bbd_options']['scales']):
        values = configs['bbd_options']['scales'].split(',')
        if len(values) == 1:
            try:
                literal_eval(values)
            except ValueError:
                errors.append('Level1Error:ScalesMustBeNoneorFloatorMultipleFloatsSeparatedbyComma')

        else:
            try:
                [float(value) for value in values]
            except ValueError:
                errors.append('Level1Error:ScalesMustBeNoneorFloatorMultipleFloatsSeparatedbyComma')

    else:
        warnings.append('Warning:NoBbdScalesSpecified')
        configs['bbd_options']['scales'] = 'None'

    if any(configs['bbd_options']['aspect_ratios_type'] in x for x in ['global', 'per predictor layer']):
        pass
    else:
        errors.append('Level1Error:NonexistentAspectRatiosType')

    if any(configs['bbd_options']['aspect_ratios']):
        try:
            ars = literal_eval(configs['bbd_options']['aspect_ratios'])
            if type(ars) is tuple:
                for ar in ars:
                    if type(ar) is tuple:
                        try:
                            [float(ar_val) for ar_val in ar]
                        except ValueError:
                            errors.append('Level1Error:AspectRatiosMustbeTupleofFloatsorTupleofTuplesofFloats')
                    else:
                        try:
                            float(ar)
                        except ValueError:
                            errors.append('Level1Error:AspectRatiosMustbeTupleofFloatsorTupleofTuplesofFloats')
                            break
            else:
                errors.append('Level1Error:AspectRatiosMustbeTupleofFloatsorTupleofTuplesofFloats')
        except ValueError:
            errors.append('Level1Error:AspectRatiosMustbeTupleofFloatsorTupleofTuplesofFloats')

    else:
        errors.append('Level1Error:AspectRatiosMustbeSpecified')

    if any(configs['bbd_options']['number_classes']):
        try:
            int(configs['bbd_options']['number_classes'])
        except ValueError:
            errors.append('Level1Error:NoNumberofBbdClassesSpecified')
    else:
        errors.append('Level1Error:NoNumberofBbdClassesSpecified')

    if any(configs['bbd_options']['steps']):
        try:
            steps = literal_eval(configs['bbd_options']['steps'])
            if type(steps) is tuple:
                for step in steps:
                    if type(step) is tuple:
                        try:
                            [float(step_val) for step_val in step]
                        except ValueError:
                            errors.append('Level1Error:StepsMustbeNoneorTupleofFloatsorTupleofTuplesofTwoFloats')
                    else:
                        try:
                            float(step)
                        except ValueError:
                            errors.append('Level1Error:StepsMustbeNoneorTupleofFloatsorTupleofTuplesofTwoFloats')
                            break
            elif steps is None:
                pass
            else:
                errors.append('Level1Error:StepsMustbeNoneorTupleofFloatsorTupleofTuplesofTwoFloats')
        except ValueError:
            errors.append('Level1Error:StepsMustbeNoneorTupleofFloatsorTupleofTuplesofTwoFloats')

    else:
        warnings.append('Warning:NoStepsSpecified')
        configs['bbd_options']['steps'] = 'None'

    if any(configs['bbd_options']['offsets']):
        try:
            offsets = literal_eval(configs['bbd_options']['offsets'])
            if type(offsets) is tuple:
                for offset in offsets:
                    if type(offset) is tuple:
                        try:
                            [float(offset_val) for offset_val in offset]
                        except ValueError:
                            errors.append('Level1Error:OffsetsMustbeNoneorTupleofFloatsorTupleofTuplesofTwoFloats')
                    else:
                        try:
                            float(offset)
                        except ValueError:
                            errors.append('Level1Error:OffsetsMustbeNoneorTupleofFloatsorTupleofTuplesofTwoFloats')
                            break
            elif offsets is None:
                pass
            else:
                errors.append('Level1Error:OffsetsMustbeNoneorTupleofFloatsorTupleofTuplesofTwoFloats')
        except ValueError:
            errors.append('Level1Error:OffsetsMustbeNoneorTupleofFloatsorTupleofTuplesofTwoFloats')

    else:
        warnings.append('Warning:NoOffsetsSpecified')
        configs['bbd_options']['offsets'] = 'None'

    if any(configs['bbd_options']['variances']):
        try:
            variances = literal_eval(configs['bbd_options']['variances'])
            if type(variances) is tuple:
                if len(variances) == 4:
                    try:
                        [float(variance) for variance in variances]
                    except ValueError:
                        errors.append('Level1Error:VariancesMustbeTupleofFourFloatsGreaterthanZero')
                else:
                    errors.append('Level1Error:VariancesMustbeTupleofFourFloatsGreaterthanZero')
            else:
                errors.append('Level1Error:VariancesMustbeTupleofFourFloatsGreaterthanZero')
        except ValueError:
            errors.append('Level1Error:VariancesMustbeTupleofFourFloatsGreaterthanZero')

    else:
        warnings.append('Warning:NoOffsetsSpecified')
        configs['bbd_options']['variances'] = '(1.0, 1.0, 1.0, 1.0)'

    if any(configs['bbd_options']['confidence_threshold']):
        try:
            float(configs['bbd_options']['confidence_threshold'])
        except ValueError:
            warnings.append('Warning:ConfidenceThresholdShouldBeFloat')
            configs['bbd_options']['confidence_threshold'] = '0.1'
    else:
        configs['bbd_options']['confidence_threshold'] = '0.1'

    if any(configs['bbd_options']['iou_threshold']):
        try:
            float(configs['bbd_options']['iou_threshold'])
        except ValueError:
            warnings.append('Warning:IoUThresholdShouldBeFloat')
            configs['bbd_options']['iou_threshold'] = '0.5'
    else:
        configs['bbd_options']['iou_threshold'] = '0.5'

    if any(configs['bbd_options']['top_k']):
        try:
            int(configs['bbd_options']['top_k'])
        except ValueError:
            warnings.append('Warning:NoBbdTopKSpecified')
            configs['bbd_options']['top_k'] = '200'
    else:
        warnings.append('Warning:NoBbdTopKSpecified')
        configs['bbd_options']['top_k'] = '200'

    if any(configs['bbd_options']['nms_maximum_output']):
        try:
            int(configs['bbd_options']['nms_maximum_output'])
        except ValueError:
            warnings.append('Warning:NoBbdNmsSpecified')
            configs['bbd_options']['nms_maximum_output'] = '400'
    else:
        warnings.append('Warning:NoBbdNmsSpecified')
        configs['bbd_options']['nms_maximum_output'] = '400'

    if any(configs['bbd_options']['coordinates_type'] in x for x in ['centroids', 'minmax', 'corners']):
        pass
    else:
        errors.append('Level1Error:NonexistentCoordinatesType')

    try:
        str2bool(configs['bbd_options']['two_boxes_for_AR1_switch'])
    except ValueError:
        warnings.append('Warning:TwoBoxesforAR1ShouldBeBool')
        configs['bbd_options']['two_boxes_for_AR1_switch'] = 'False'

    try:
        str2bool(configs['bbd_options']['clip_boxes_switch'])
    except ValueError:
        warnings.append('Warning:ClipBoxesShouldBeBool')
        configs['bbd_options']['clip_boxes_switch'] = 'False'

    try:
        str2bool(configs['bbd_options']['normalize_coordinates_switch'])
    except ValueError:
        warnings.append('Warning:NormalizeCoordinatesShouldBeBool')
        configs['bbd_options']['normalize_coordinates_switch'] = 'False'

    if any(configs['bbd_options']['positive_iou_threshold']):
        try:
            float(configs['bbd_options']['positive_iou_threshold'])
        except ValueError:
            warnings.append('Warning:PositiveIoUThresholdShouldBeFloat')
            configs['bbd_options']['positive_iou_threshold'] = '0.5'
    else:
        configs['bbd_options']['positive_iou_threshold'] = '0.5'

    if any(configs['bbd_options']['negative_iou_limit']):
        try:
            float(configs['bbd_options']['negative_iou_limit'])
        except ValueError:
            warnings.append('Warning:NegativeIoULimitShouldBeFloat')
            configs['bbd_options']['negative_iou_limit'] = '0.3'
    else:
        configs['bbd_options']['negative_iou_limit'] = '0.3'

    return configs, errors, warnings


def level_two_error_checking(configs):
    engine_configs = EngineConfigurations(configs)
    errors = engine_configs.train_data.errors\
             + engine_configs.val_data.errors\
             + engine_configs.test_data.errors\
             + engine_configs.saver.errors
    warnings = engine_configs.train_data.warnings\
               + engine_configs.val_data.warnings\
               + engine_configs.test_data.warnings\
               + engine_configs.saver.warnings

    return engine_configs, errors, warnings


def get_io(layer_definitions):
    inner_skip_starts = []
    outer_skip_starts = []
    bbd_hooks = []
    errors = []
    inputs = None
    x = None

    for i, layer_definition in enumerate(layer_definitions):
        # try:
            layer = create_layer(layer_definition)

            if i == 0:
                if layer.type != 'Input':
                    errors.append('Level3Error:FirstLayerMustBeInput')
                    break

                else:
                    inputs = layer.keras_layer

            elif i == 1:
                # try:
                    if layer.type in ['Xception', 'VGG16', 'VGG19', 'ResNet50', 'ResNet101', 'ResNet152',
                                      'ResNet50V2', 'ResNet101V2', 'ResNet152V2', 'ResNeXt50', 'ResNeXt101',
                                      'InceptionV3', 'InceptionResNetV2', 'DenseNet121', 'DenseNet169',
                                      'DenseNet201', 'MobileNet', 'MobileNetV2']:
                        inputs = layer.keras_layer.input
                        x = layer.keras_layer.output
                        if literal_eval(layer.include_skips):
                            outer_skip_starts = layer.skips
                        if literal_eval(layer.include_hooks):
                            bbd_hooks = layer.hooks
                    else:
                        x = layer.keras_layer(inputs)
                # except:
                #     errors.append('Level3Error:CouldNotAdd ' + layer.type + ' AsALayer')

            elif layer.type == 'Resize convolution 2D' or layer.type == 'Resize convolution 3D':
                # try:
                    x = layer.keras_upsample_layer(x)
                    x = layer.keras_conv_layer(x)
                # except:
                #     errors.append('Level3Error:CouldNotAdd ' + layer.type + ' AsALayer')

            elif layer.type == 'Outer skip source':
                # try:
                    outer_skip_starts.append(x)
                # except:
                #     errors.append('Level3Error:CouldNotAdd ' + layer.type + ' AsALayer')

            elif layer.type in ['Xception', 'VGG16', 'VGG19', 'ResNet50', 'ResNet101', 'ResNet152',
                                'ResNet50V2', 'ResNet101V2', 'ResNet152V2', 'ResNeXt50', 'ResNeXt101',
                                'InceptionV3', 'InceptionResNetV2', 'DenseNet121', 'DenseNet169',
                                'DenseNet201', 'MobileNet', 'MobileNetV2']:
                # try:
                    inputs = layer.keras_layer.input
                    x = layer.keras_layer.output
                    if literal_eval(layer.include_skips):
                        outer_skip_starts = layer.skips
                    if literal_eval(layer.include_hooks):
                        bbd_hooks = layer.hooks
                # except:
                #     errors.append('Level3Error:CouldNotAdd ' + layer.type + ' AsALayer')

            elif layer.type == 'Outer skip target':
                # try:
                    if layer.skip_type == 'concatenate':
                        x = keras.layers.Concatenate()([outer_skip_starts[-1], x])

                    if layer.skip_type == 'add':
                        x = keras.layers.Add()([outer_skip_starts[-1], x])

                    if layer.skip_type == 'subtract':
                        x = keras.layers.Subtract()([outer_skip_starts[-1], x])

                    if layer.skip_type == 'multiply':
                        x = keras.layers.Multiply()([outer_skip_starts[-1], x])

                    if layer.skip_type == 'average':
                        x = keras.layers.Average()([outer_skip_starts[-1], x])

                    if layer.skip_type == 'maximum':
                        x = keras.layers.Maximum()([outer_skip_starts[-1], x])

                    outer_skip_starts.pop()

                # except:
                #     errors.append('Level3Error:CouldNotAdd ' + layer.type + ' AsALayer')

            elif layer.type == 'Inner skip source':
                # try:
                    inner_skip_starts.append(x)
                # except:
                #     errors.append('Level3Error:CouldNotAdd ' + layer.type + ' AsALayer')

            elif layer.type == 'Inner skip target':
                # try:
                    if layer.skip_type == 'concatenate':
                        x = keras.layers.Concatenate()([inner_skip_starts[0], x])

                    if layer.skip_type == 'add':
                        x = keras.layers.Add()([inner_skip_starts[0], x])

                    if layer.skip_type == 'subtract':
                        x = keras.layers.Subtract()([inner_skip_starts[0], x])

                    if layer.skip_type == 'multiply':
                        x = keras.layers.Multiply()([inner_skip_starts[0], x])

                    if layer.skip_type == 'average':
                        x = keras.layers.Average()([inner_skip_starts[0], x])

                    if layer.skip_type == 'maximum':
                        x = keras.layers.Maximum()([inner_skip_starts[0], x])

                    inner_skip_starts.pop()
                # except:
                #     errors.append('Level3Error:CouldNotAdd ' + layer.type + ' AsALayer')

            elif layer.type == 'Hook connection source':
                # try:
                    bbd_hooks.append(x)
                # except:
                #     errors.append('Level3Error:CouldNotAdd ' + layer.type + ' AsALayer')

            else:
                # try:
                    x = layer.keras_layer(x)
                # except:
                #     errors.append('Level3Error:CouldNotAdd ' + layer.type + ' AsALayer')

        # except:
        #     errors.append('Level3Error:CouldNotCreateLayerFromLayerSpecifications')

    return inputs, x, bbd_hooks, errors


def get_cgan_d_io(layer_definitions, gen_input):
    inner_skip_starts = []
    outer_skip_starts = []
    errors = []
    inputs = None
    x = None

    for i, layer_definition in enumerate(layer_definitions):
        try:
            layer = create_layer(layer_definition)

            if i == 0:
                if layer.type != 'Input':
                    errors.append('Level3Error:FirstLayerMustBeInput')
                    break

                else:
                    source_layer = create_layer(gen_input)
                    source = source_layer.keras_layer
                    target = layer.keras_layer
                    inputs = Concatenate(axis=-1)([target, source])

            elif i == 1:
                try:
                    if layer.type in ['Xception', 'VGG16', 'VGG19', 'ResNet50', 'ResNet101', 'ResNet152',
                                      'ResNet50V2', 'ResNet101V2', 'ResNet152V2', 'ResNeXt50', 'ResNeXt101',
                                      'InceptionV3', 'InceptionResNetV2', 'DenseNet121', 'DenseNet169',
                                      'DenseNet201', 'MobileNet', 'MobileNetV2']:
                        inputs = layer.keras_layer.input
                        x = layer.keras_layer.output
                        if literal_eval(layer.include_skips):
                            outer_skip_starts = layer.skips
                    else:
                        x = layer.keras_layer(inputs)
                except:
                    errors.append('Level3Error:CouldNotAdd ' + layer.type + ' AsALayer')

            elif layer.type == 'Resize convolution 2D' or layer.type == 'Resize convolution 3D':
                try:
                    x = layer.keras_upsample_layer(x)
                    x = layer.keras_conv_layer(x)
                except:
                    errors.append('Level3Error:CouldNotAdd ' + layer.type + ' AsALayer')

            elif layer.type == 'Outer skip source':
                try:
                    outer_skip_starts.append(x)
                except:
                    errors.append('Level3Error:CouldNotAdd ' + layer.type + ' AsALayer')

            elif layer.type in ['Xception', 'VGG16', 'VGG19', 'ResNet50', 'ResNet101', 'ResNet152',
                                'ResNet50V2', 'ResNet101V2', 'ResNet152V2', 'ResNeXt50', 'ResNeXt101',
                                'InceptionV3', 'InceptionResNetV2', 'DenseNet121', 'DenseNet169',
                                'DenseNet201', 'MobileNet', 'MobileNetV2']:
                try:
                    inputs = layer.keras_layer.input
                    x = layer.keras_layer.output
                    if literal_eval(layer.include_skips):
                        outer_skip_starts = layer.skips
                except:
                    errors.append('Level3Error:CouldNotAdd ' + layer.type + ' AsALayer')

            elif layer.type == 'Outer skip target':
                try:
                    if layer.skip_type == 'concatenate':
                        x = keras.layers.Concatenate()([outer_skip_starts[-1], x])

                    if layer.skip_type == 'add':
                        x = keras.layers.Add()([outer_skip_starts[-1], x])

                    if layer.skip_type == 'subtract':
                        x = keras.layers.Subtract()([outer_skip_starts[-1], x])

                    if layer.skip_type == 'multiply':
                        x = keras.layers.Multiply()([outer_skip_starts[-1], x])

                    if layer.skip_type == 'average':
                        x = keras.layers.Average()([outer_skip_starts[-1], x])

                    if layer.skip_type == 'maximum':
                        x = keras.layers.Maximum()([outer_skip_starts[-1], x])

                    outer_skip_starts.pop()

                except:
                    errors.append('Level3Error:CouldNotAdd ' + layer.type + ' AsALayer')

            elif layer.type == 'Inner skip source':
                try:
                    inner_skip_starts.append(x)
                except:
                    errors.append('Level3Error:CouldNotAdd ' + layer.type + ' AsALayer')

            elif layer.type == 'Inner skip target':
                try:
                    if layer.skip_type == 'concatenate':
                        x = keras.layers.Concatenate()([inner_skip_starts[0], x])

                    if layer.skip_type == 'add':
                        x = keras.layers.Add()([inner_skip_starts[0], x])

                    if layer.skip_type == 'subtract':
                        x = keras.layers.Subtract()([inner_skip_starts[0], x])

                    if layer.skip_type == 'multiply':
                        x = keras.layers.Multiply()([inner_skip_starts[0], x])

                    if layer.skip_type == 'average':
                        x = keras.layers.Average()([inner_skip_starts[0], x])

                    if layer.skip_type == 'maximum':
                        x = keras.layers.Maximum()([inner_skip_starts[0], x])

                    inner_skip_starts.pop()
                except:
                    errors.append('Level3Error:CouldNotAdd ' + layer.type + ' AsALayer')

            else:
                try:
                    x = layer.keras_layer(x)
                except:
                    errors.append('Level3Error:CouldNotAdd ' + layer.type + ' AsALayer')

        except:
            errors.append('Level3Error:CouldNotCreateLayerFromLayerSpecifications')

    return [target, source], x, errors
