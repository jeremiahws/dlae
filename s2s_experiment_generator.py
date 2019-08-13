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
"""s2s_experiment_generator.py

Experiment generator for SeedNet2SatNet studies. This experiment generator
iterates over the hyperparameters of the sliding window as well as different
CNN constructs and training parameters.
"""


from src.utils.general_utils import load_config
from src.utils.engine_utils import level_one_error_checking, level_two_error_checking
from src.engine.constructor import Dlae
import argparse
import itertools
import os


def main(FLAGS):
    configs = load_config(FLAGS.base_configs)
    configs['training_configurations']['batch_size'] = '512'

    c_train_X = 'train_classification_windows'
    c_train_y = 'train_classification_labels'
    c_valid_X = 'valid_classification_windows'
    c_valid_y = 'valid_classification_labels'
    l_train_X = 'train_localization_windows'
    l_train_y = 'train_localization_labels'
    l_valid_X = 'valid_localization_windows'
    l_valid_y = 'valid_localization_labels'

    paddings = FLAGS.padding.split(',')
    window_sizes = FLAGS.window_size.split(',')
    strides = FLAGS.stride.split(',')
    bg2sat_ratios = FLAGS.bg2sat_ratio.split(',')
    experiments = [window_sizes, strides, paddings, bg2sat_ratios]

    for experiment in itertools.product(*experiments):
        if int(experiment[0]) - 2 * int(experiment[2]) >= FLAGS.minimum_center\
                and int(experiment[1]) < int(experiment[0]) - 2 * int(experiment[2]):
            h5_path_append = '_seedNet2satNet_windowsize_{}_stride_{}_padding_{}_ratio_{}_trainfraction_{}.h5'.format(experiment[0], experiment[1], experiment[2], experiment[3], FLAGS.train_fraction)
            csv_path_append = '_seedNet2satNet_windowsize_{}_stride_{}_padding_{}_ratio_{}_trainfraction_{}.csv'.format(experiment[0], experiment[1], experiment[2], experiment[3], FLAGS.train_fraction)

            create_command = 'python C:/Users/jsanders/Desktop/seedNet2satNet/create_annotated_windows.py '\
                           + '--satnet_data_dir={}'.format(FLAGS.satnet_data_dir)\
                           + '--save_data_dir={}'.format(FLAGS.save_data_dir)\
                           + '--train_file_names={}'.format(FLAGS.train_file_names)\
                           + '--valid_file_names={}'.format(FLAGS.valid_file_names)\
                           + '--window_size={}'.format(FLAGS.window_size)\
                           + '--stride={}'.format(FLAGS.stride)\
                           + '--padding={}'.format(FLAGS.padding)\
                           + '--bg2sat_ratio={}'.format(FLAGS.bg2sat_ratio)\
                           + '--format={}'.format(FLAGS.format)

            os.system(create_command)

            configs['paths']['train_X'] = os.path.join(FLAGS.satnet_data_dir, c_train_X + h5_path_append)
            configs['paths']['train_y'] = os.path.join(FLAGS.satnet_data_dir, c_train_y + h5_path_append)
            configs['paths']['validation_X'] = os.path.join(FLAGS.satnet_data_dir, c_valid_X + h5_path_append)
            configs['paths']['validation_y'] = os.path.join(FLAGS.satnet_data_dir, c_valid_y + h5_path_append)
            configs['monitors']['accuracy_switch'] = 'True'
            configs['monitors']['mse_switch'] = 'False'
            configs['save_configurations']['save_model_path'] = os.path.join(FLAGS.save_model_path, 'classification_model' + h5_path_append)
            configs['save_configurations']['save_csv_path'] = os.path.join(FLAGS.save_csv_path, 'classification_csv' + csv_path_append)
            configs['save_configurations']['save_checkpoints_path'] = os.path.join(FLAGS.save_ckpt_path, 'classification_ckpt' + h5_path_append)

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
                    engine = Dlae(configs).run()
                    if any(engine.errors):
                        print('Level 3 errors encountered.')
                        print("Please fix the level 3 errors below before continuing:")
                        for error in engine.errors:
                            print(error)

            os.remove(configs['paths']['train_X'])
            os.remove(configs['paths']['train_y'])
            os.remove(configs['paths']['validation_X'])
            os.remove(configs['paths']['validation_y'])

            configs['paths']['train_X'] = os.path.join(FLAGS.satnet_data_dir, l_train_X + h5_path_append)
            configs['paths']['train_y'] = os.path.join(FLAGS.satnet_data_dir, l_train_y + h5_path_append)
            configs['paths']['validation_X'] = os.path.join(FLAGS.satnet_data_dir, l_valid_X + h5_path_append)
            configs['paths']['validation_y'] = os.path.join(FLAGS.satnet_data_dir, l_valid_y + h5_path_append)
            configs['preprocessing']['categorical_switch'] = 'False'
            configs['preprocessing']['weight_loss_switch'] = 'False'
            configs['loss_function']['loss'] = 'mean_squared_error'
            configs['monitors']['accuracy_switch'] = 'False'
            configs['monitors']['mse_switch'] = 'True'
            configs['save_configurations']['save_model_path'] = os.path.join(FLAGS.save_model_path, 'localization_model' + h5_path_append)
            configs['save_configurations']['save_csv_path'] = os.path.join(FLAGS.save_csv_path, 'localization_csv' + csv_path_append)
            configs['save_configurations']['save_checkpoints_path'] = os.path.join(FLAGS.save_ckpt_path, 'localization_ckpt' + h5_path_append)
            layers = configs['layers']['serial_layer_list']
            layers.pop()
            configs['layers']['serial_layer_list'] = layers

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
                    engine = Dlae(configs).run()
                    if any(engine.errors):
                        print('Level 3 errors encountered.')
                        print("Please fix the level 3 errors below before continuing:")
                        for error in engine.errors:
                            print(error)

            os.remove(configs['paths']['train_X'])
            os.remove(configs['paths']['train_y'])
            os.remove(configs['paths']['validation_X'])
            os.remove(configs['paths']['validation_y'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--base_configs', type=str,
                        default='/home/jsanders/Desktop/github/dlae_migration2/dlae/configs/seedNet2satNet_classifier_experiment_gen.json',
                        help='Template configuration file to make spawns from.')

    parser.add_argument('--window_size', type=str,
                        default='32',
                        help='Size of sub-windows (in pixels); single value or multiple values separated by a comma.')

    parser.add_argument('--stride', type=str,
                        default='3',
                        help='Stride of the sliding window (in pixels); single value or multiple values separated by a comma.')

    parser.add_argument('--padding', type=str,
                        default='12',
                        help='Padding to apply to sub-windows to avoid edge cases (in pixels); single value or multiple values separated by a comma.')

    parser.add_argument('--width', type=int,
                        default=512,
                        help='Width of the image (in pixels).')

    parser.add_argument('--height', type=int,
                        default=512,
                        help='Height of the image (in pixels).')

    parser.add_argument('--bg2sat_ratio', type=str,
                        default='10',
                        help='Ratio of background:satellite sub-windows in the training dataset.')

    parser.add_argument('--train_fraction', type=float,
                        default=1.0,
                        help='Fraction of total number of training images curated.')

    parser.add_argument('--satnet_data_dir', type=str,
                        default='/opt/tfrecords/SatNet.v.1.0.0.0/SatNet/data',
                        help='Top level directory for SatNet data from all sensors and collection days.')

    parser.add_argument('--save_data_dir', type=str,
                        default='/home/jsanders/Desktop/data/seedNet2satNet',
                        help='Directory where to save the sub-window data.')

    parser.add_argument('--train_file_names', type=str,
                        default='/opt/tfrecords/SatNet.v.1.0.0.0/SatNet/info/data_split/train.txt',
                        help='Path to .txt file containing training file names.')

    parser.add_argument('--valid_file_names', type=str,
                        default='/opt/tfrecords/SatNet.v.1.0.0.0/SatNet/info/data_split/valid.txt',
                        help='Path to .txt file containing validation file names.')

    parser.add_argument('--save_ckpt_path', type=str,
                        default='/home/jsanders/Desktop/github/dlae_migration2/dlae/ckpt',
                        help='Path to save the model checkpoints.')

    parser.add_argument('--save_model_path', type=str,
                        default='/home/jsanders/Desktop/github/dlae_migration2/dlae/models',
                        help='Path to save the models.')

    parser.add_argument('--save_csv_path', type=str,
                        default='/home/jsanders/Desktop/github/dlae_migration2/dlae/csv',
                        help='Path to save the CSV logs.')

    parser.add_argument('--minimum_center', type=int,
                        default=4,
                        help='Minimum of the central coverage of the sub-windows.')

    parser.add_argument('--format', type=str,
                        default='hdf5',
                        help='File format to save images and annotations in (hdf or tfrecords).')

    # parse known arguements
    FLAGS, unparsed = parser.parse_known_args()

    main(FLAGS)
