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
"""fcn_experiment_generator.py

Experiment generator for generating deep learning experiments
for a multi-class image segmentation task.
"""


from src.utils.general_utils import load_config
from src.utils.engine_utils import level_one_error_checking, level_two_error_checking
from src.engine.constructor import Dlae
import argparse
import itertools
import os
from ast import literal_eval


def main(FLAGS):
    # set GPU devices to use
    # os.environ['CUDA_VISIBLE_DEVICES'] = '6,7'

    # define the experiments
    encoders = FLAGS.encoders.split(',')
    losses = FLAGS.losses.split(',')
    alpha = FLAGS.loss_param1.split(',')
    experiments = [encoders, losses, alpha]
    print(experiments)
    for experiment in itertools.product(*experiments):
        print(experiment)

    for experiment in itertools.product(*experiments):
        # switch to activate training session
        do_train = True

        # load the base configurations
        if experiment[0] == 'UNet':
            configs = load_config(os.path.join(FLAGS.base_configs_dir, 'unet2d.json'))
        elif experiment[0] == 'VGG16' or experiment[0] == 'VGG19':
            configs = load_config(os.path.join(FLAGS.base_configs_dir, 'vgg16_unet.json'))
        else:
            configs = load_config(os.path.join(FLAGS.base_configs_dir, 'xception_unet.json'))

        # apply some augmentation
        configs['augmentation']['apply_augmentation_switch'] = 'False'
        #configs['augmentation']['width_shift'] = '0.15'
        #configs['augmentation']['height_shift'] = '0.15'
        configs['augmentation']['rotation_range'] = '3'
        configs['augmentation']['zoom_range'] = '0.15'
        configs['augmentation']['shear_range'] = '0.05'

        # perform some preprocessing
        configs['preprocessing']['categorical_switch'] = 'True'
        configs['preprocessing']['minimum_image_intensity'] = '0.0'
        configs['preprocessing']['maximum_image_intensity'] = '2048.0'
        configs['preprocessing']['categories'] = '{}'.format(FLAGS.classes)
        configs['preprocessing']['normalization_type'] = '{}'.format(FLAGS.normalization)

        # set the training configurations
        if FLAGS.num_gpus > 1:
            configs['training_configurations']['hardware'] = 'multi-gpu'
        else:
            configs['training_configurations']['hardware'] = 'gpu'
        configs['training_configurations']['number_of_gpus'] = '{}'.format(FLAGS.num_gpus)
        configs['training_configurations']['batch_size'] = '{}'.format(FLAGS.batch_size)
        configs['training_configurations']['epochs'] = '{}'.format(FLAGS.epochs)
        configs['training_configurations']['validation_split'] = '0.0'

        # set the learning rate
        configs['learning_rate_schedule']['learning_rate'] = '{}'.format(FLAGS.learning_rate)

        # turn on savers
        configs['save_configurations']['save_checkpoints_switch'] = 'True'
        configs['save_configurations']['save_csv_switch'] = 'True'

        # define data paths
        configs['paths']['train_X'] = FLAGS.train_X_path
        configs['paths']['train_y'] = FLAGS.train_y_path
        configs['paths']['validation_X'] = FLAGS.valid_X_path
        configs['paths']['validation_y'] = FLAGS.valid_y_path

        ckpt_path_append = 'encoder_{}_loss_{}_alpha_{}_beta_{}_ckpt.h5'.format(experiment[0],
                                                                                experiment[1],
                                                                                experiment[2],
                                                                                1. - literal_eval(experiment[2]))
        csv_path_append = 'encoder_{}_loss_{}_alpha_{}_beta_{}_history.csv'.format(experiment[0],
                                                                                   experiment[1],
                                                                                   experiment[2],
                                                                                   1. - literal_eval(experiment[2]))

        configs['save_configurations']['save_csv_path'] = os.path.join(FLAGS.save_csv_path, csv_path_append)
        configs['save_configurations']['save_checkpoints_path'] = os.path.join(FLAGS.save_ckpt_path, ckpt_path_append)

        layers = configs['layers']['serial_layer_list']
        input = layers[0]
        input_parts = input.split(':')
        input_parts[-1] = '({}, {}, {})'.format(FLAGS.height, FLAGS.width, FLAGS.channels)
        input = ':'.join(input_parts)
        configs['config_file']['input_shape'] = '({}, {}, {})'.format(FLAGS.height, FLAGS.width, FLAGS.channels)

        if experiment[0] == 'UNet':
            layers[0] = input
            last_conv = layers[-3]
            last_conv_parts = last_conv.split(':')
            last_conv_parts[1] = '{}'.format(FLAGS.classes)
            last_conv = ':'.join(last_conv_parts)
            layers[-3] = last_conv
            if FLAGS.use_skip_connections:
                pass
            else:
                while 'Outer skip target:concatenate' in layers:
                    layers.remove('Outer skip target:concatenate')
            print(layers)
            configs['layers']['serial_layer_list'] = layers
        else:
            encoder = layers[1]
            decoder = layers[2:]
            last_conv = decoder[-3]
            last_conv_parts = last_conv.split(':')
            last_conv_parts[1] = '{}'.format(FLAGS.classes)
            last_conv = ':'.join(last_conv_parts)
            decoder[-3] = last_conv
            encoder_parts = encoder.split(':')
            encoder_parts[0] = experiment[0]
            encoder_parts[3] = '({}, {}, {})'.format(FLAGS.height, FLAGS.width, FLAGS.channels)
            if FLAGS.use_skip_connections:
                encoder_parts[-2] = 'True'
            else:
                encoder_parts[-2] = 'False'
                while 'Outer skip target:concatenate' in decoder:
                    decoder.remove('Outer skip target:concatenate')
            encoder = ':'.join(encoder_parts)

            if FLAGS.use_imagenet_weights:
                encoder_parts[1] = 'True'
                encoder_parts[2] = 'imagenet'
                configs['preprocessing']['repeat_X_switch'] = 'True'
                configs['preprocessing']['repeat_X_quantity'] = '3'
            else:
                encoder_parts[1] = 'False'
                encoder_parts[2] = 'none'
                configs['preprocessing']['repeat_X_switch'] = 'False'

            # inject some layers to connect the encoder to the decoder
            # for this experiment not many injectors are needed, but may be for others
            # they're broken up by the type of encoder
            if experiment[0] == 'VGG16' or experiment[0] == 'VGG19':
                pass
            elif experiment[0] == 'DenseNet121' or experiment[0] == 'DenseNet169' or experiment[0] == 'DenseNet201':
                pass
            elif experiment[0] == 'InceptionResNetV2' or experiment[0] == 'InceptionV3':
                decoder.insert(0, 'Zero padding 2D:((1, 1), (1, 1))')
            elif experiment[0] == 'MobileNet' or experiment[0] == 'MobileNetV2':
                pass
            elif experiment[0] == 'ResNet50' or experiment[0] == 'ResNet101' or experiment[0] == 'ResNet152':
                pass
            elif experiment[0] == 'ResNet50V2' or experiment[0] == 'ResNet101V2' or experiment[0] == 'ResNet152V2':
                pass
            elif experiment[0] == 'ResNeXt50' or experiment[0] == 'ResNeXt101':
                pass
            elif experiment[0] == 'Xception':
                pass
            else:
                print('Invalid encoder type:', experiment[0])
                do_train = False

            layers = []
            layers.extend([input, encoder])
            for layer in decoder: layers.append(layer)
            configs['layers']['serial_layer_list'] = layers

        # ensure xentropy/jaccard/focal only used once per encoder
        if experiment[1] == 'sparse_categorical_crossentropy'\
                or experiment[1] == 'categorical_crossentropy'\
                or experiment[1] == 'jaccard'\
                or experiment[1] == 'focal':
            if experiment[1] == 'focal':
                configs['loss_function']['parameter1'] = '0.75'
                configs['loss_function']['parameter2'] = '2.0'
            if experiment[1] == 'jaccard':
                configs['loss_function']['parameter1'] = '100.0'
            if experiment[2] == '0.3':
                configs['loss_function']['loss'] = experiment[1]
            else:
                do_train = False
        elif experiment[1] == 'tversky':
            configs['loss_function']['loss'] = experiment[1]
            configs['loss_function']['parameter1'] = experiment[2]
            configs['loss_function']['parameter2'] = str(1. - literal_eval(experiment[2]))
        else:
            do_train = False

        if do_train is True:
            configs_lvl1, errors_lvl1, warnings_lvl1 = level_one_error_checking(configs)

            if any(warnings_lvl1):
                with open('errors.txt', 'a') as f:
                    for warning in warnings_lvl1:
                        f.write("%s\n" % warning)
                    f.close()
                    print('Level 1 warnings encountered.')
                    print("The following level 1 warnings were identified and corrected based on engine defaults:")
                    for warning in warnings_lvl1:
                        print(warning)

            if any(errors_lvl1):
                print('Level 1 errors encountered.')
                print("Please fix the level 1 errors below before continuing:")
                for error in errors_lvl1:
                    print(error)
            else:
                configs_lvl2, errors_lvl2, warnings_lvl2 = level_two_error_checking(configs_lvl1)

                if any(warnings_lvl2):
                    print('Level 2 warnings encountered.')
                    print("The following level 2 warnings were identified and corrected based on engine defaults:")
                    for warning in warnings_lvl2:
                        print(warning)

                if any(errors_lvl2):
                    print('Level 2 errors encountered.')
                    print("Please fix the level 2 errors below before continuing:")
                    for error in errors_lvl2:
                        print(error)
                else:
                    engine = Dlae(configs)
                    engine.run()
                    if any(engine.errors):
                        print('Level 3 errors encountered.')
                        print("Please fix the level 3 errors below before continuing:")
                        for error in engine.errors:
                            print(error)
        else:
            pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--base_configs_dir', type=str,
                        default=r'C:\Users\jsanders\Desktop\dlae\src\prebuilt_configs',
                        help='Directory to template configuration files from which to make spawns from.')

    parser.add_argument('--losses', type=str,
                        default='sparse_categorical_crossentropy',
                        help='Loss functions to investigate; single value or multiple values separated by a comma.')

    parser.add_argument('--encoders', type=str,
                        default='VGG16,VGG19,DenseNet121,DenseNet169,DenseNet201,Xception,'
                              + 'ResNet50,ResNet101,ResNet152,ResNet50V2,ResNet101V2,ResNet152V2,'
                              + 'ResNeXt50,ResNeXt101,InceptionResNetV2,InceptionV3',
                        help='Convolutional encoders to investigate; single value or multiple values separated by a comma.')

    parser.add_argument('--use_skip_connections', type=bool,
                        default=True,
                        help='Whether or not to use skip connections between the encoder and decoder.')

    parser.add_argument('--use_imagenet_weights', type=bool,
                        default=False,
                        help='Whether or not to use a warm start with ImageNet weights.')

    parser.add_argument('--width', type=int,
                        default=512,
                        help='Width of the image (in pixels).')

    parser.add_argument('--height', type=int,
                        default=512,
                        help='Height of the image (in pixels).')

    parser.add_argument('--channels', type=int,
                        default=1,
                        help='Number of image channels.')

    parser.add_argument('--classes', type=int,
                        default=1,
                        help='Number of object classes.')

    parser.add_argument('--normalization', type=str,
                        default='samplewise_unity_x',
                        help='Type of image normalization.')

    parser.add_argument('--batch_size', type=int,
                        default=1,
                        help='Batch size for training.')

    parser.add_argument('--epochs', type=int,
                        default=1000,
                        help='Number of training epochs.')

    parser.add_argument('--learning_rate', type=float,
                        default=0.0001,
                        help='Learning rate.')

    parser.add_argument('--num_gpus', type=int,
                        default=1,
                        help='Number of GPUs to use (greedy utilization).')

    parser.add_argument('--loss_param1', type=str,
                        default='0.3,0.5,0.7',
                        help='Parameter 1 of the loss function(s).')

    parser.add_argument('--train_X_path', type=str,
                        default=r'C:\Users\jsanders\Desktop\dlae\datasets\example_fcn_X_data.h5',
                        help='NF1 patient training images.')

    parser.add_argument('--train_y_path', type=str,
                        default=r'C:\Users\jsanders\Desktop\dlae\datasets\example_fcn_y_data.h5',
                        help='NF1 patient training masks.')

    parser.add_argument('--valid_X_path', type=str,
                        default=r'C:\Users\jsanders\Desktop\dlae\datasets\example_fcn_X_data.h5',
                        help='NF1 patient validation images.')

    parser.add_argument('--valid_y_path', type=str,
                        default=r'C:\Users\jsanders\Desktop\dlae\datasets\example_fcn_y_data.h5',
                        help='NF1 patient validation masks.')

    parser.add_argument('--save_ckpt_path', type=str,
                        default=r'C:\Users\jsanders\Desktop\dlae\ckpt',
                        help='Path to save the model checkpoints.')

    parser.add_argument('--save_csv_path', type=str,
                        default=r'C:\Users\jsanders\Desktop\dlae\csv',
                        help='Path to save the CSV logs.')

    # parse known arguements
    FLAGS, unparsed = parser.parse_known_args()

    # launch experiments
    main(FLAGS)
