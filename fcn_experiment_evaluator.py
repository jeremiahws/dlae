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
"""fcn_experiment_evaluator.py

Experiment evaluator for the segmentation experiments produced by the
experiment generator in fcn_experiment_generator.py
"""


from src.utils.general_utils import load_config, read_hdf5, read_hdf5_multientry, write_hdf5
from src.utils.engine_utils import level_one_error_checking, level_two_error_checking
from src.engine.constructor import Dlae
import argparse
import itertools
import os
from ast import literal_eval
import numpy as np
from glob import glob
import math
from medpy import metric as mpm
from sklearn import metrics as skm
import pandas as pd


def relative_volume_difference(A, B):
    '''Compute relative volume difference between two segmentation masks.
    The voxel size gets canceled out in the division and is therefore not
    a required input.

    :param A: (binary numpy array) reference segmentaton mask
    :param B: (binary numpy array) predicted segmentaton mask
    :return: relative volume difference
    '''

    volume_A = np.sum(A)
    volume_B = np.sum(B)
    rvd = (volume_A - volume_B) / volume_A

    return rvd


def modified_hausdorff_distance(A, B):
    '''Compute modified Hausdorff distance between two segmentation masks.

    :param A: (numpy array) reference segmentaton mask
    :param B: (numpy array) predicted segmentaton mask
    :return: modified Hausdorff distance
    '''

    shape_A = A.shape
    shape_B = B.shape

    fwd = 0
    for i in range(shape_A[0]):
        min_dist = math.inf
        for j in range(shape_B[0]):
            new_dist = np.linalg.norm(A[i]-B[j])
            if new_dist < min_dist:
                min_dist = new_dist

        fwd = fwd + min_dist
    fwd = fwd / shape_A[0]

    rvd = 0
    for j in range(shape_B[0]):
        min_dist = math.inf
        for i in range(shape_A[0]):
            new_dist = np.linalg.norm(A[i] - B[j])
            if new_dist < min_dist:
                min_dist = new_dist

        rvd = rvd + min_dist
    rvd = rvd / shape_B[0]

    mhd = np.max([fwd, rvd])

    return mhd


def main(FLAGS):
    # set GPU device to use
    # os.environ['CUDA_VISIBLE_DEVICES'] = '6'

    # define the experiments
    encoders = FLAGS.encoders.split(',')
    losses = FLAGS.losses.split(',')
    alpha = FLAGS.loss_param1.split(',')
    experiments = [encoders, losses, alpha]
    print(experiments)

    # define data path
    image_files = os.listdir(FLAGS.test_X_dir)
    anno_files = os.listdir(FLAGS.test_y_dir)

    for experiment in itertools.product(*experiments):
        # switch to activate training session
        do_eval = True

        # load the base configurations
        if experiment[0] == 'UNet':
            configs = load_config(os.path.join(FLAGS.base_configs_dir, 'unet2d.json'))
        elif experiment[0] == 'UNet3D':
            configs = load_config(os.path.join(FLAGS.base_configs_dir, 'unet3d.json'))
        elif experiment[0] == 'VGG16' or experiment[0] == 'VGG19':
            configs = load_config(os.path.join(FLAGS.base_configs_dir, 'vgg16_unet.json'))
        else:
            configs = load_config(os.path.join(FLAGS.base_configs_dir, 'xception_unet.json'))

        # set the path to the model (checkpoints are fine for this)
        ckpt_name = 'encoder_{}_loss_{}_alpha_{}_beta_{}_ckpt.h5'.format(experiment[0],
                                                                         experiment[1],
                                                                         experiment[2],
                                                                         1. - literal_eval(experiment[2]))
        configs['paths']['load_model'] = os.path.join(FLAGS.ckpt_dir, ckpt_name)

        # switch to inference
        configs['config_file']['type_signal'] = 'Inference'

        # perform some preprocessing
        configs['preprocessing']['categorical_switch'] = 'True'
        configs['preprocessing']['minimum_image_intensity'] = '0.0'
        configs['preprocessing']['maximum_image_intensity'] = '2048.0'
        configs['preprocessing']['normalization_type'] = '{}'.format(FLAGS.normalization)

        # set some other configurations
        configs['training_configurations']['batch_size'] = '{}'.format(FLAGS.batch_size)
        configs['config_file']['input_shape'] = '({}, {}, {})'.format(FLAGS.height, FLAGS.width, FLAGS.channels)

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
                do_eval = False
        elif experiment[1] == 'tversky':
            configs['loss_function']['loss'] = experiment[1]
            configs['loss_function']['parameter1'] = experiment[2]
            configs['loss_function']['parameter2'] = str(1. - literal_eval(experiment[2]))
        else:
            do_eval = False

        # create a location to store evaluation metrics
        metrics = np.zeros((len(image_files), FLAGS.classes, 8))
        overall_accuracy = np.zeros((len(image_files),))

        # create a file writer to store the metrics
        excel_name = '{}_{}_{}_{}_metrics.xlsx'.format(experiment[0],
                                                       experiment[1],
                                                       experiment[2],
                                                       1. - literal_eval(experiment[2]))
        writer = pd.ExcelWriter(excel_name)

        for i in range(len(image_files)):
            # define path to the test data
            configs['paths']['test_X'] = os.path.join(FLAGS.test_X_dir, image_files[i])

            if do_eval is True:
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

                pred_file = glob(os.path.join(FLAGS.predictions_dir, '*.h5'))[0]
                pt_name = image_files[i].split('.')[0]
                new_name_raw = pt_name + '_{}_{}_{}_{}_raw.h5'.format(experiment[0],
                                                                      experiment[1],
                                                                      experiment[2],
                                                                      1. - literal_eval(experiment[2]))
                new_path = os.path.join(os.path.split(pred_file)[0], 'predictions', 'prostate_mri_segmentation')
                new_file_raw = os.path.join(new_path, new_name_raw)
                os.rename(pred_file, new_file_raw)

                preds = read_hdf5(new_file_raw)
                preds = np.argmax(preds, axis=-1)

                ref = read_hdf5_multientry(os.path.join(FLAGS.test_y_dir, anno_files[i]))
                ref = np.squeeze(np.asarray(ref))

                overall_accuracy[i] = skm.accuracy_score(ref.flatten(), preds.flatten())
                # print('acc:', skm.accuracy_score(ref.flatten(), preds.flatten()))
                for j in range(FLAGS.classes):
                    organ_pred = (preds == j).astype(np.int64)
                    organ_ref = (ref == j).astype(np.int64)
                    if np.sum(organ_pred) == 0 or np.sum(organ_ref) == 0:
                        metrics[i, j, 0] = 0.
                        metrics[i, j, 1] = 0.
                        metrics[i, j, 2] = 1.
                        metrics[i, j, 3] = 0.
                        metrics[i, j, 4] = 0.
                        metrics[i, j, 5] = 0.
                        metrics[i, j, 6] = np.inf
                        metrics[i, j, 7] = np.inf
                    else:
                        metrics[i, j, 0] = skm.jaccard_score(organ_ref.flatten(), organ_pred.flatten(), average='binary')
                        metrics[i, j, 1] = skm.f1_score(organ_ref.flatten(), organ_pred.flatten(), average='binary')
                        metrics[i, j, 2] = relative_volume_difference(organ_ref, organ_pred)
                        metrics[i, j, 3] = skm.precision_score(organ_ref.flatten(), organ_pred.flatten())
                        metrics[i, j, 4] = skm.recall_score(organ_ref.flatten(), organ_pred.flatten())
                        metrics[i, j, 5] = skm.matthews_corrcoef(organ_ref.flatten(), organ_pred.flatten())
                        metrics[i, j, 6] = mpm.hd95(organ_pred, organ_ref)
                        metrics[i, j, 7] = mpm.assd(organ_pred, organ_ref)
                        # print('ji:',skm.jaccard_score(organ_ref.flatten(), organ_pred.flatten(), average='binary'))
                        # print('dsc:',skm.f1_score(organ_ref.flatten(), organ_pred.flatten(), average='binary'))
                        # print('rvd:',relative_volume_difference(organ_ref, organ_pred))
                        # print('p:',skm.precision_score(organ_ref.flatten(), organ_pred.flatten()))
                        # print('r:',skm.recall_score(organ_ref.flatten(), organ_pred.flatten()))
                        # print('mcc:',skm.matthews_corrcoef(organ_ref.flatten(), organ_pred.flatten()))
                        # print('hd95:',mpm.hd95(organ_pred, organ_ref))
                        # print('assd:',mpm.assd(organ_pred, organ_ref))
                print(overall_accuracy[i])
                print(metrics[i])

            else:
                pass

        if do_eval is True:
            for k in range(metrics.shape[-1]):
                data = pd.DataFrame(metrics[:, :, k], columns=['bg', 'pros', 'eus', 'sv', 'rect', 'blad'])
                data.to_excel(writer, sheet_name=str(k))
            acc = pd.DataFrame(overall_accuracy, columns=['acc'])
            acc.to_excel(writer, sheet_name='acc')
            writer.save()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--base_configs_dir', type=str,
                        default=r'C:\Users\JSanders1\Desktop\Github\dlae\prebuilt_configs',
                        help='Directory to template configuration files from which to make spawns from.')

    parser.add_argument('--losses', type=str,
                        default='focal,jaccard,categorical_crossentropy,tversky',
                        help='Loss functions to investigate; single value or multiple values separated by a comma.')

    parser.add_argument('--encoders', type=str,
                        default='UNet,DenseNet121',
                        help='Convolutional encoders to investigate; single value or multiple values separated by a comma.')

    parser.add_argument('--width', type=int,
                        default=512,
                        help='Width of the image (in pixels).')

    parser.add_argument('--height', type=int,
                        default=512,
                        help='Height of the image (in pixels).')

    parser.add_argument('--slices', type=int,
                        default=1,
                        help='Number of slices (for 3D volume inputs).')

    parser.add_argument('--channels', type=int,
                        default=1,
                        help='Number of image channels.')

    parser.add_argument('--classes', type=int,
                        default=6,
                        help='Number of object classes.')

    parser.add_argument('--normalization', type=str,
                        default='samplewise_unity_x',
                        help='Type of image normalization.')

    parser.add_argument('--batch_size', type=int,
                        default=1,
                        help='Batch size for training.')

    parser.add_argument('--loss_param1', type=str,
                        default='0.3,0.5,0.7',
                        help='Parameter 1 of the loss function(s).')

    parser.add_argument('--test_X_dir', type=str,
                        default=r'C:\Users\JSanders1\Desktop\Github\dlae\datasets\brachy_mri_images',
                        help='Path to directory storing the testing images.')

    parser.add_argument('--test_y_dir', type=str,
                        default=r'C:\Users\JSanders1\Desktop\Github\dlae\datasets\brachy_mri_annos',
                        help='Path to directory storing testing annotations for computing performance metrics.')

    parser.add_argument('--predictions_dir', type=str,
                        default=r'C:\Users\JSanders1\Desktop\Github\dlae',
                        help='Path to directory storing model predictions. By default, top level directory of DLAE.')

    parser.add_argument('--ckpt_dir', type=str,
                        default=r'C:\Users\JSanders1\Desktop\Github\dlae\ckpt',
                        help='Path to where the model checkpoints are stored.')

    # parse known arguements
    FLAGS, unparsed = parser.parse_known_args()

    # launch experiments
    main(FLAGS)
