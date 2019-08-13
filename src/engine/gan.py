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
"""src/engine/gan.py

Constructs the generative adversarial network technique of DLAE.
"""


from math import floor
import tensorflow as tf
from src.utils.engine_utils import *
import numpy as np
import datetime
import time
from src.utils.general_utils import write_hdf5
import imageio
import os
import csv


class GenerativeAdversairalNetwork(object):
    def __init__(self, engine_configs):
        self.engine_configs = engine_configs
        self.graph = tf.get_default_graph()
        self.errors = []
        self.generator1 = None
        self.generator2 = None
        self.discriminator1 = None
        self.discriminator2 = None
        self.gan = None

        self.parallel_generator1 = None
        self.parallel_generator2 = None
        self.parallel_discriminator1 = None
        self.parallel_discriminator2 = None
        self.parallel_gan = None

    def construct_graph(self):
        if self.engine_configs.train_options.s_hardware == "cpu"\
                or self.engine_configs.train_options.s_hardware == "multi-gpu":
            device = "/cpu:0"
        else:
            device = "/gpu:0"

        if self.engine_configs.loss_function.loss == "pix2pix":
            with tf.device(device):
                if any(self.engine_configs.layers.s_listOfGeneratorLayers):
                    gen_inputs, gen_outputs, _, gen_errors = get_io(self.engine_configs.layers.s_listOfGeneratorLayers)
                    if any(gen_errors):
                        [self.errors.append(error) for error in gen_errors]
                    else:
                        try:
                            self.generator1 = keras.models.Model(inputs=gen_inputs, outputs=gen_outputs)
                        except:
                            self.errors.append('Level3Error:CouldNotConstructGanGeneratorGraph')

                else:
                    self.errors.append('Level3Error:NoListofGeneratorLayersFound')

                if any(self.engine_configs.layers.s_listOfDiscriminatorLayers):
                    discrim_inputs, discrim_outputs, discrim_errors = get_cgan_d_io(self.engine_configs.layers.s_listOfDiscriminatorLayers,
                                                                                    self.engine_configs.layers.s_listOfGeneratorLayers[0])
                    if any(discrim_errors):
                        [self.errors.append(error) for error in discrim_errors]
                    else:
                        try:
                            self.discriminator1 = keras.models.Model(inputs=discrim_inputs, outputs=discrim_outputs)
                        except:
                            self.errors.append('Level3Error:CouldNotConstructGanDiscriminatorGraph')

                else:
                    self.errors.append('Level3Error:NoListofDiscriminatorLayersFound')

        elif self.engine_configs.loss_function.loss == "cyclegan":
            with tf.device(device):
                if any(self.engine_configs.layers.s_listOfGeneratorLayers):
                    gen_inputs_Xy, gen_outputs_Xy, _, gen_errors_Xy = get_io(self.engine_configs.layers.s_listOfGeneratorLayers)
                    if any(gen_errors_Xy):
                        [self.errors.append(error) for error in gen_errors_Xy]
                    else:
                        try:
                            self.generator1 = keras.models.Model(inputs=gen_inputs_Xy, outputs=gen_outputs_Xy)
                        except:
                            self.errors.append('Level3Error:CouldNotConstructGanGenerator1Graph')

                else:
                    self.errors.append('Level3Error:NoListofGeneratorLayersFound')

                if any(self.engine_configs.layers.s_listOfGeneratorLayers):
                    gen_inputs_yX, gen_outputs_yX, _, gen_errors_yX = get_io(self.engine_configs.layers.s_listOfGeneratorLayers)
                    if any(gen_errors_yX):
                        [self.errors.append(error) for error in gen_errors_yX]
                    else:
                        try:
                            self.generator2 = keras.models.Model(inputs=gen_inputs_yX, outputs=gen_outputs_yX)
                        except:
                            self.errors.append('Level3Error:CouldNotConstructGanGenerator2Graph')

                else:
                    self.errors.append('Level3Error:NoListofGeneratorLayersFound')

                if any(self.engine_configs.layers.s_listOfDiscriminatorLayers):
                    discrim_inputs_Xy, discrim_outputs_Xy, _, discrim_errors_Xy = get_io(self.engine_configs.layers.s_listOfDiscriminatorLayers)
                    if any(discrim_errors_Xy):
                        [self.errors.append(error) for error in discrim_errors_Xy]
                    else:
                        try:
                            self.discriminator1 = keras.models.Model(inputs=discrim_inputs_Xy, outputs=discrim_outputs_Xy)
                        except:
                            self.errors.append('Level3Error:CouldNotConstructGanDiscriminator1Graph')

                else:
                    self.errors.append('Level3Error:NoListofDiscriminatorLayersFound')

                if any(self.engine_configs.layers.s_listOfDiscriminatorLayers):
                    discrim_inputs_yX, discrim_outputs_yX, _, discrim_errors_yX = get_io(self.engine_configs.layers.s_listOfDiscriminatorLayers)
                    if any(discrim_errors_yX):
                        [self.errors.append(error) for error in discrim_errors_yX]
                    else:
                        try:
                            self.discriminator2 = keras.models.Model(inputs=discrim_inputs_yX,
                                                                     outputs=discrim_outputs_yX)
                        except:
                            self.errors.append('Level3Error:CouldNotConstructGanDiscriminator2Graph')

                else:
                    self.errors.append('Level3Error:NoListofDiscriminatorLayersFound')

    def compile_graph(self):
        if self.engine_configs.loss_function.loss == "pix2pix":
            if self.engine_configs.train_options.i_nGpus > 1:
                raise NotImplementedError
            else:
                self.discriminator1.compile(optimizer=self.engine_configs.optimizer.d_optimizer,
                                            loss=keras.losses.mean_squared_error,
                                            metrics=['accuracy'])
                source = create_layer(self.engine_configs.layers.s_listOfGeneratorLayers[0])
                target = create_layer(self.engine_configs.layers.s_listOfDiscriminatorLayers[0])
                fake_image = self.generator1(source.keras_layer)
                self.discriminator1.trainable = False
                validity = self.discriminator1([fake_image, source.keras_layer])
                self.gan = keras.models.Model(inputs=[target.keras_layer, source.keras_layer],
                                              outputs=[validity, fake_image])
                self.gan.compile(optimizer=self.engine_configs.optimizer.gan_optimizer,
                                 loss=[keras.losses.mean_squared_error, keras.losses.mean_absolute_error],
                                 loss_weights=[1, self.engine_configs.loss_function.f_parameter1])

        elif self.engine_configs.loss_function.loss == "cyclegan":
            if self.engine_configs.train_options.i_nGpus > 1:
                raise NotImplementedError
            else:
                self.discriminator1.compile(optimizer=self.engine_configs.optimizer.d_optimizer,
                                            loss=keras.losses.mean_squared_error,
                                            metrics=['accuracy'])
                self.discriminator2.compile(optimizer=self.engine_configs.optimizer.d_optimizer,
                                            loss=keras.losses.mean_squared_error,
                                            metrics=['accuracy'])
                gen_input_Xy = create_layer(self.engine_configs.layers.s_listOfGeneratorLayers[0])
                gen_input_yX = create_layer(self.engine_configs.layers.s_listOfDiscriminatorLayers[0])
                gen_image_Xy = self.generator1(gen_input_Xy.keras_layer)
                gen_image_yX = self.generator2(gen_input_yX.keras_layer)
                recon_X = self.generator2(gen_image_Xy)
                recon_y = self.generator1(gen_image_yX)

                self.discriminator1.trainable = False
                self.discriminator2.trainable = False

                discrim_validity_yX = self.discriminator1(gen_image_yX)
                discrim_validity_Xy = self.discriminator2(gen_image_Xy)

                identity_X = self.generator2(gen_input_Xy.keras_layer)
                identity_y = self.generator1(gen_input_yX.keras_layer)
                self.gan = keras.models.Model(inputs=[gen_input_Xy.keras_layer, gen_input_yX.keras_layer],
                                              outputs=[discrim_validity_yX, discrim_validity_Xy,
                                                       recon_X, recon_y,
                                                       identity_X, identity_y])
                self.gan.compile(loss=[keras.losses.mean_squared_error, keras.losses.mean_squared_error,
                                       keras.losses.mean_absolute_error, keras.losses.mean_absolute_error,
                                       keras.losses.mean_absolute_error, keras.losses.mean_absolute_error],
                                 loss_weights=[1, 1, self.engine_configs.loss_function.f_parameter1,
                                               self.engine_configs.loss_function.f_parameter1,
                                               self.engine_configs.loss_function.f_parameter2,
                                               self.engine_configs.loss_function.f_parameter2],
                                 optimizer=self.engine_configs.optimizer.gan_optimizer)

    def train_graph(self):
        # Check if there is any validation data
        if self.engine_configs.val_data.valX is not None:
            # There is some validation data, move on
            pass
        else:
            if self.engine_configs.train_options.f_validationSplit > 0.0:
                val_split = self.engine_configs.train_options.f_validationSplit
                n_tot_imgs = self.engine_configs.train_data.trainX.shape[0]
                n_val_imgs = floor(val_split * n_tot_imgs)
                self.engine_configs.val_data.valX = self.engine_configs.train_data.trainX[:n_val_imgs]
                self.engine_configs.val_data.valY = self.engine_configs.train_data.trainY[:n_val_imgs]
                self.engine_configs.train_data.trainX = self.engine_configs.train_data.trainX[n_val_imgs:]
                self.engine_configs.train_data.trainY = self.engine_configs.train_data.trainY[n_val_imgs:]
            else:
                # No validation data
                pass

        # Ignore last training images if rem(#images, batch_size) > 0
        n_batches = floor(self.engine_configs.train_data.trainX.shape[0] / self.engine_configs.train_options.i_batchSize)

        if self.engine_configs.loss_function.loss == "pix2pix":
            with self.graph.as_default():
                if self.engine_configs.train_options.i_nGpus > 1:
                    # see: https://gist.github.com/joelthchao/ef6caa586b647c3c032a4f84d52e3a11
                    raise NotImplementedError
                else:
                    # configure labels for patchgan discriminator
                    patch_size = self.discriminator1.output_shape
                    patch_size = patch_size[1:-1]
                    patch_size = (patch_size + (1,))
                    valid = np.ones((self.engine_configs.train_options.i_batchSize,) + patch_size)
                    fake = np.zeros((self.engine_configs.train_options.i_batchSize,) + patch_size)

                    for epoch in range(self.engine_configs.train_options.i_epochs):
                        for index in range(n_batches):
                            batch_X = self.engine_configs.train_data.trainX[index * self.engine_configs.train_options.i_batchSize:(index + 1) * self.engine_configs.train_options.i_batchSize]
                            batch_y = self.engine_configs.train_data.trainY[index * self.engine_configs.train_options.i_batchSize:(index + 1) * self.engine_configs.train_options.i_batchSize]

                            if np.random.random() < 0.5:
                                for ind in range(batch_X.shape[0]):
                                    flip_X = np.fliplr(batch_X[ind])
                                    flip_y = np.fliplr(batch_y[ind])
                                    batch_X[ind] = flip_X
                                    batch_y[ind] = flip_y

                            fake_image = self.generator1.predict(batch_X)

                            d_loss_real = self.discriminator1.train_on_batch([batch_y, batch_X], valid)
                            d_loss_fake = self.discriminator1.train_on_batch([fake_image, batch_X], fake)
                            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                            g_loss = self.gan.train_on_batch([batch_y, batch_X], [valid, batch_y])
                            print("epoch {}/{}".format(epoch + 1, self.engine_configs.train_options.i_epochs),
                                  "batch {}/{}".format(index + 1, n_batches),
                                  "g_loss:", g_loss[0], "d_loss:", d_loss[0], "d_acc:", 100*d_loss[1])

                        if self.engine_configs.val_data.valX is not None and self.engine_configs.loss_function.image_context == '2D':
                            ind = np.random.randint(self.engine_configs.val_data.valX.shape[0])
                            val_img = self.engine_configs.val_data.valX[ind]
                            val_img = np.expand_dims(val_img, axis=0)
                            preds = self.generator1.predict(val_img, batch_size=1)

                            stamp = datetime.datetime.fromtimestamp(time.time()).strftime('date_%Y_%m_%d_time_%H_%M_%S')
                            cwd = os.getcwd()
                            img_name = os.path.join(cwd, "temp/epoch_{}_".format(epoch) + stamp + ".png")

                            if preds.shape[-1] == 1:
                                preds = np.squeeze(preds)
                                preds = np.expand_dims(preds, axis=-1)
                            else:
                                preds = np.squeeze(preds)

                            img_pair = np.concatenate([self.engine_configs.val_data.valY[ind], preds], axis=1)

                            try:
                                imageio.imwrite(img_name, (0.5 * img_pair + 0.5)*255.0)
                            except:
                                if img_pair.shape[-1] != 1 or img_pair.shape[-1] != 3:
                                    try:
                                        imageio.imwrite(img_name, (0.5 * img_pair[:, :, 0] + 0.5)*255)
                                    except:
                                        pass
                                else:
                                    pass

                        if self.engine_configs.saver.b_saveCkpt is True:
                            parts = os.path.split(self.engine_configs.saver.s_saveCkptPath)
                            file_name = parts[-1]
                            new_file_name = 'epoch_' + str(epoch) + '_cktp_pix2pix_generator_' + file_name
                            new_path = os.path.join(parts[0], new_file_name)
                            self.generator1.save(new_path)

                        if self.engine_configs.saver.b_saveCSV is True:
                            if epoch == 0:
                                with open(self.engine_configs.saver.s_saveCSVPath, 'w', newline='') as f:
                                    fieldnames = ['epoch', 'g_loss', 'd_loss', 'd_acc']
                                    writer = csv.DictWriter(f, fieldnames=fieldnames)

                                    writer.writeheader()
                                    writer.writerow({'epoch': epoch, 'g_loss': g_loss[0],
                                                     'd_loss': d_loss[0], 'd_acc': 100*d_loss[1]})
                                    f.close()
                            else:
                                with open(self.engine_configs.saver.s_saveCSVPath, 'a', newline='') as f:
                                    fieldnames = ['epoch', 'g_loss', 'd_loss', 'd_acc']
                                    writer = csv.DictWriter(f, fieldnames=fieldnames)

                                    writer.writerow({'epoch': epoch, 'g_loss': g_loss[0],
                                                     'd_loss': d_loss[0], 'd_acc': 100*d_loss[1]})
                                    f.close()

                    if self.engine_configs.saver.b_saveModel is True:
                        path = os.path.join(os.path.dirname(self.engine_configs.saver.s_saveModelPath),
                                            "pix2pix_generator_" + os.path.basename(self.engine_configs.saver.s_saveModelPath))
                        self.generator1.save(path)

        elif self.engine_configs.loss_function.loss == "cyclegan":
            with self.graph.as_default():
                if self.engine_configs.train_options.i_nGpus > 1:
                    # see: https://gist.github.com/joelthchao/ef6caa586b647c3c032a4f84d52e3a11
                    raise NotImplementedError
                else:
                    # configure labels for patchgan discriminator
                    patch_size = self.discriminator1.output_shape
                    patch_size = patch_size[1:-1]
                    patch_size = (patch_size + (1,))
                    valid = np.ones((self.engine_configs.train_options.i_batchSize,) + patch_size)
                    fake = np.zeros((self.engine_configs.train_options.i_batchSize,) + patch_size)

                    for epoch in range(self.engine_configs.train_options.i_epochs):
                        inds = np.random.permutation(self.engine_configs.train_data.trainX.shape[0])
                        self.engine_configs.train_data.trainX = self.engine_configs.train_data.trainX[inds]
                        self.engine_configs.train_data.trainY = self.engine_configs.train_data.trainY[inds]
                        for index in range(n_batches):
                            batch_X = self.engine_configs.train_data.trainX[index * self.engine_configs.train_options.i_batchSize:(index + 1) * self.engine_configs.train_options.i_batchSize]
                            batch_y = self.engine_configs.train_data.trainY[index * self.engine_configs.train_options.i_batchSize:(index + 1) * self.engine_configs.train_options.i_batchSize]

                            if np.random.random() < 0.5:
                                for ind in range(batch_X.shape[0]):
                                    flip_X = np.fliplr(batch_X[ind])
                                    flip_y = np.fliplr(batch_y[ind])
                                    batch_X[ind] = flip_X
                                    batch_y[ind] = flip_y

                            Xy = self.generator1.predict(batch_X)
                            yX = self.generator2.predict(batch_y)

                            dX_loss_real = self.discriminator1.train_on_batch(batch_X, valid)
                            dX_loss_fake = self.discriminator1.train_on_batch(yX, fake)
                            dX_loss = 0.5 * np.add(dX_loss_real, dX_loss_fake)

                            dy_loss_real = self.discriminator2.train_on_batch(batch_y, valid)
                            dy_loss_fake = self.discriminator2.train_on_batch(Xy, fake)
                            dy_loss = 0.5 * np.add(dy_loss_real, dy_loss_fake)

                            d_loss = 0.5 * np.add(dX_loss, dy_loss)

                            g_loss = self.gan.train_on_batch([batch_X, batch_y],
                                                             [valid, valid, batch_X, batch_y, batch_X, batch_y])

                            print("epoch {}/{}".format(epoch + 1, self.engine_configs.train_options.i_epochs),
                                  "batch {}/{}".format(index + 1, n_batches),
                                  "g_loss:", g_loss[0], "g_adv:", np.mean(g_loss[1:3]),
                                  "g_recon:", np.mean(g_loss[3:5]), "g_identity:", np.mean(g_loss[5:6]),
                                  "d_loss:", d_loss[0], "d_acc:", 100 * d_loss[1])

                        if self.engine_configs.val_data.valX is not None\
                                and self.engine_configs.loss_function.image_context == '2D':
                            ind = np.random.randint(self.engine_configs.val_data.valX.shape[0])
                            val_img1 = self.engine_configs.val_data.valX[ind]
                            val_img1 = np.expand_dims(val_img1, axis=0)
                            val_img2 = self.engine_configs.val_data.valY[ind]
                            val_img2 = np.expand_dims(val_img2, axis=0)
                            preds1 = self.generator1.predict(val_img1, batch_size=1)
                            preds2 = self.generator2.predict(val_img2, batch_size=1)
                            recon1 = self.generator2.predict(preds1)
                            recon2 = self.generator1.predict(preds2)

                            stamp = datetime.datetime.fromtimestamp(time.time()).strftime('date_%Y_%m_%d_time_%H_%M_%S')
                            cwd = os.getcwd()
                            img_name = os.path.join(cwd, "temp/epoch_{}_".format(epoch) + stamp + ".png")

                            if preds1.shape[-1] == 1:
                                preds1 = np.squeeze(preds1)
                                preds1 = np.expand_dims(preds1, axis=-1)
                            else:
                                preds1 = np.squeeze(preds1)

                            if preds2.shape[-1] == 1:
                                preds2 = np.squeeze(preds2)
                                preds2 = np.expand_dims(preds2, axis=-1)
                            else:
                                preds2 = np.squeeze(preds2)

                            if recon1.shape[-1] == 1:
                                recon1 = np.squeeze(recon1)
                                recon1 = np.expand_dims(recon1, axis=-1)
                            else:
                                recon1 = np.squeeze(recon1)

                            if recon2.shape[-1] == 1:
                                recon2 = np.squeeze(recon2)
                                recon2 = np.expand_dims(recon2, axis=-1)
                            else:
                                recon2 = np.squeeze(recon2)

                            img_pair1 = np.concatenate([self.engine_configs.val_data.valX[ind], preds1, recon1], axis=1)
                            img_pair2 = np.concatenate([self.engine_configs.val_data.valY[ind], preds2, recon2], axis=1)
                            img_pairs = np.concatenate([img_pair1, img_pair2], axis=0)

                            try:
                                imageio.imwrite(img_name, (0.5 * img_pairs + 0.5)*255.0)
                            except:
                                if img_pairs.shape[-1] != 1 or img_pairs.shape[-1] != 3:
                                    try:
                                        imageio.imwrite(img_name, (0.5 * img_pairs[:, :, 0] + 0.5)*255)
                                    except:
                                        pass
                                else:
                                    pass

                        if self.engine_configs.saver.b_saveCkpt is True:
                            parts = os.path.split(self.engine_configs.saver.s_saveCkptPath)
                            file_name = parts[-1]
                            new_file_name1 = 'epoch_' + str(epoch) + '_cktp_cyclegan_generator1_' + file_name
                            new_path1 = os.path.join(parts[0], new_file_name1)
                            new_file_name2 = 'epoch_' + str(epoch) + '_cktp_cyclegan_generator2_' + file_name
                            new_path2 = os.path.join(parts[0], new_file_name2)
                            self.generator1.save(new_path1)
                            self.generator2.save(new_path2)

                        if self.engine_configs.saver.b_saveCSV is True:
                            if epoch == 0:
                                with open(self.engine_configs.saver.s_saveCSVPath, 'w', newline='') as f:
                                    fieldnames = ['epoch', 'g_loss', 'g_adv', 'g_recon',
                                                  'g_identity', 'd_loss', 'd_acc']
                                    writer = csv.DictWriter(f, fieldnames=fieldnames)

                                    writer.writeheader()
                                    writer.writerow({'epoch': epoch, 'g_loss': g_loss[0],
                                                     'g_adv': np.mean(g_loss[1:3]), 'g_recon': np.mean(g_loss[3:5]),
                                                     'g_identity': np.mean(g_loss[5:6]), 'd_loss': d_loss[0],
                                                     'd_acc': 100*d_loss[1]})
                                    f.close()
                            else:
                                with open(self.engine_configs.saver.s_saveCSVPath, 'a', newline='') as f:
                                    fieldnames = ['epoch', 'g_loss', 'g_adv', 'g_recon',
                                                  'g_identity', 'd_loss', 'd_acc']
                                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                                    writer.writerow({'epoch': epoch, 'g_loss': g_loss[0],
                                                     'g_adv': np.mean(g_loss[1:3]), 'g_recon': np.mean(g_loss[3:5]),
                                                     'g_identity': np.mean(g_loss[5:6]), 'd_loss': d_loss[0],
                                                     'd_acc': 100 * d_loss[1]})
                                    f.close()

                    if self.engine_configs.saver.b_saveModel is True:
                        path1 = os.path.join(os.path.dirname(self.engine_configs.saver.s_saveModelPath),
                                             "cyclegan_generator1_" + os.path.basename(self.engine_configs.saver.s_saveModelPath))
                        path2 = os.path.join(os.path.dirname(self.engine_configs.saver.s_saveModelPath),
                                             "cyclegan_generator2_" + os.path.basename(self.engine_configs.saver.s_saveModelPath))
                        self.generator1.save(path1)
                        self.generator2.save(path2)

    def retrain_graph(self):
        raise NotImplementedError

    def predict_on_graph(self):
        try:
            self.model = keras.models.load_model(self.engine_configs.loader.s_loadModelPath)
            predictions = self.model.predict(self.engine_configs.test_data.testX)

            stamp = datetime.datetime.fromtimestamp(time.time()).strftime('date_%Y_%m_%d_time_%H_%M_%S')

            write_hdf5('gan_predictions_' + stamp + '.h5', predictions)
        except:
            self.errors.append('Level3Error:CouldNotMakePredictionsonGanGraph')
