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
"""metnet_seg_experiment_evaluator.py

Experiment evaluator for the metNet segmentation experiments.
"""


from src.utils.general_utils import read_hdf5, read_hdf5_multientry, write_hdf5
import argparse
import os
import numpy as np
from glob import glob
from medpy import metric as mpm
from sklearn import metrics as skm
import pandas as pd
from math import sqrt
import keras.backend.tensorflow_backend as K
from src.utils.data_generators import FCN2DDatasetGenerator
import time
import datetime
from keras.models import load_model


def relative_volume_difference(A, B):
    '''Compute relative volume difference between two segmentation masks.
    The voxel size gets canceled out in the division and is therefore not
    a required input.

    :param A: (binary numpy array) reference segmentaton mask
    :param B: (binary numpy array) predicted segmentaton mask
    :return: relative volume difference
    '''

    volume_A = int(np.sum(A))
    volume_B = int(np.sum(B))
    rvd = (volume_A - volume_B) / volume_A

    return rvd


def jaccard_index(A, B):
    '''Compute Jaccard index (IoU) between two segmentation masks.

    :param A: (numpy array) reference segmentaton mask
    :param B: (numpy array) predicted segmentaton mask
    :return: Jaccard index
    '''

    both = np.logical_and(A, B)
    either = np.logical_or(A, B)
    ji = int(np.sum(both)) / int(np.sum(either))

    return ji


def dice_similarity_coefficient(A, B):
    '''Compute Dice similarity coefficient between two segmentation masks.

    :param A: (numpy array) reference segmentaton mask
    :param B: (numpy array) predicted segmentaton mask
    :return: Dice similarity coefficient
    '''

    both = np.logical_and(A, B)
    dsc = 2 * int(np.sum(both)) / (int(np.sum(A)) + int(np.sum(B)))

    return dsc


def precision(A, B):
    '''Compute precision between two segmentation masks.

    :param A: (numpy array) reference segmentaton mask
    :param B: (numpy array) predicted segmentaton mask
    :return: precision
    '''

    tp = int(np.sum(np.logical_and(A, B)))
    fp = int(np.sum(np.logical_and(B, np.logical_not(A))))
    p = tp / (tp + fp)

    return p


def recall(A, B):
    '''Compute recall between two segmentation masks.

    :param A: (numpy array) reference segmentaton mask
    :param B: (numpy array) predicted segmentaton mask
    :return: recall
    '''

    tp = int(np.sum(np.logical_and(A, B)))
    fn = int(np.sum(np.logical_and(A, np.logical_not(B))))
    r = tp / (tp + fn)

    return r


def matthews_correlation_coefficient(A, B):
    '''Compute Matthews correlation coefficient between two segmentation masks.

    :param A: (numpy array) reference segmentaton mask
    :param B: (numpy array) predicted segmentaton mask
    :return: Matthews correlation coefficient
    '''

    tp = int(np.sum(np.logical_and(A, B)))
    fp = int(np.sum(np.logical_and(B, np.logical_not(A))))
    tn = int(np.sum(np.logical_and(np.logical_not(A), np.logical_not(B))))
    fn = int(np.sum(np.logical_and(A, np.logical_not(B))))
    mcc = (tp * tn - fp * fn) / (sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))

    return mcc


def main(FLAGS):
    # set GPU device to use
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'

    # get the models to evaluate
    ckpts = glob(os.path.join(FLAGS.ckpt_dir, '*.h5'))

    # get data files and sort them
    image_files = os.listdir(FLAGS.test_X_dir)
    anno_files = os.listdir(FLAGS.test_y_dir)
    image_files.sort()
    anno_files.sort()

    for ckpt in ckpts:
        K.clear_session()

        # set model to load
        ckpt_name = os.path.basename(ckpt)

        # create a location to store evaluation metrics
        metrics = np.zeros((len(image_files), FLAGS.classes, 8))
        overall_accuracy = np.zeros((len(image_files),))

        # create a file writer to store the metrics
        excel_name = os.path.splitext(os.path.basename(ckpt))[0] + '.xlsx'
        writer = pd.ExcelWriter(excel_name)

        model = load_model(ckpt)

        for i in range(len(image_files)):
            # define path to the test data
            test_path = os.path.join(FLAGS.test_X_dir, image_files[i])

            generator = FCN2DDatasetGenerator(test_path,
                                              batch_size=FLAGS.batch_size,
                                              subset='test',
                                              normalization=FLAGS.normalization,
                                              categorical_labels=True,
                                              num_classes=FLAGS.classes)

            # check if the images and annotations are the correct files
            print(image_files[i], anno_files[i])

            preds = model.predict_generator(generator.generate(), steps=len(generator))
            stamp = datetime.datetime.fromtimestamp(time.time()).strftime('date_%Y_%m_%d_time_%H_%M_%S')
            write_hdf5('fcn_predictions_' + stamp + '.h5', preds)

            pred_file = glob(os.path.join(FLAGS.predictions_temp_dir, '*.h5'))[0]
            pt_name = image_files[i].split('.')[0]
            new_name_raw = pt_name + ckpt_name
            new_file_raw = os.path.join(FLAGS.predictions_final_dir, new_name_raw)
            os.rename(pred_file, new_file_raw)

            ref = read_hdf5_multientry(os.path.join(FLAGS.test_y_dir, anno_files[i]))
            ref = np.squeeze(np.asarray(ref))

            preds = read_hdf5(new_file_raw)
            preds = np.argmax(preds, axis=-1)

            overall_accuracy[i] = skm.accuracy_score(ref.flatten(), preds.flatten())
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
                    metrics[i, j, 0] = jaccard_index(organ_ref, organ_pred)
                    metrics[i, j, 1] = dice_similarity_coefficient(organ_ref, organ_pred)
                    metrics[i, j, 2] = relative_volume_difference(organ_ref, organ_pred)
                    metrics[i, j, 3] = precision(organ_ref, organ_pred)
                    metrics[i, j, 4] = recall(organ_ref, organ_pred)
                    metrics[i, j, 5] = matthews_correlation_coefficient(organ_ref, organ_pred)
                    metrics[i, j, 6] = mpm.hd95(organ_pred, organ_ref)
                    metrics[i, j, 7] = mpm.assd(organ_pred, organ_ref)
            print(overall_accuracy[i])
            print(metrics[i])

        for k in range(metrics.shape[-1]):
            data = pd.DataFrame(metrics[:, :, k], columns=['bg', 'met'])
            data.to_excel(writer, sheet_name=str(k))
        acc = pd.DataFrame(overall_accuracy, columns=['acc'])
        acc.to_excel(writer, sheet_name='acc')
        writer.save()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

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

    parser.add_argument('--test_X_dir', type=str,
                        default=r'C:\Users\JSanders1\Desktop\Github\dlae\datasets\brachy_mri_images',
                        help='Path to directory storing the testing images.')

    parser.add_argument('--test_y_dir', type=str,
                        default=r'C:\Users\JSanders1\Desktop\Github\dlae\datasets\brachy_mri_annos',
                        help='Path to directory storing testing annotations for computing performance metrics.')

    parser.add_argument('--predictions_temp_dir', type=str,
                        default=r'C:\Users\JSanders1\Desktop\Github\dlae',
                        help='Path to directory storing model predictions. By default, top level directory of DLAE.')

    parser.add_argument('--predictions_final_dir', type=str,
                        default=r'C:\Users\JSanders1\Desktop\Github\dlae\predictions\prostate_mri_segmentation',
                        help='Path to directory storing model predictions. By default, top level directory of DLAE.')

    parser.add_argument('--ckpt_dir', type=str,
                        default=r'C:\Users\JSanders1\Desktop\Github\dlae\ckpt',
                        help='Path to where the model checkpoints are stored.')

    # parse known arguements
    FLAGS, unparsed = parser.parse_known_args()

    # launch experiments
    main(FLAGS)
