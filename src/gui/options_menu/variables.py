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
"""src/gui/options_menu/variables.py"""


import tkinter as tk


class OptionsMenuVariables(object):
    def __init__(self):
        ##############################################################################
        # Variables for loss submenu
        ##############################################################################
        self.o_loss = ('categorical_crossentropy', 'sparse_categorical_crossentropy', 'mean_squared_error', 'mean_absolute_error', 'tversky', 'pix2pix', 'cyclegan', 'ssd')
        self.s_loss = tk.StringVar(value=self.o_loss[0])
        self.s_loss_param1 = tk.StringVar(value="0.0")
        self.s_loss_param2 = tk.StringVar(value="0.0")

        ##############################################################################
        # Variables for learning rate schedule submenu
        ##############################################################################
        self.s_base_lr = tk.StringVar(value="0.0001")
        self.s_lr_decay = tk.StringVar(value="0.0")
        self.bool_decay_on_plateau = tk.BooleanVar(value=True)
        self.s_decay_on_plateau_factor = tk.StringVar(value="0.2")
        self.s_decay_on_plateau_patience = tk.StringVar(value="3")
        self.bool_step_decay = tk.BooleanVar(value=False)
        self.s_step_decay_factor = tk.StringVar(value="0.1")
        self.s_step_decay_period = tk.StringVar(value="3")
        self.s_d_lr = tk.StringVar(value="0.0001:0.0")
        self.s_gan_lr = tk.StringVar(value="0.0001:0.0")

        ##############################################################################
        # Variables for optimizer submenu
        ##############################################################################
        self.o_optimizer = ('Adam', 'NAdam', 'SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adamax')
        self.s_optimizer = tk.StringVar(value=self.o_optimizer[0])
        self.s_optimizer_beta1 = tk.StringVar(value="0.9")
        self.s_optimizer_beta2 = tk.StringVar(value="0.999")
        self.s_optimizer_rho = tk.StringVar(value="0.9")
        self.s_optimizer_momentum = tk.StringVar(value="0.0")
        self.s_optimizer_epsilon = tk.StringVar(value="None")
        self.s_d_optimizer = tk.StringVar(value="Adam:0.9:0.999:0.9:0.0:None")
        self.s_gan_optimizer = tk.StringVar(value="Adam:0.9:0.999:0.9:0.0:None")

        ##############################################################################
        # Variables for training configurations submenu
        ##############################################################################
        self.o_hardware = ('gpu', 'multi-gpu', 'cpu')
        self.s_hardware = tk.StringVar(value=self.o_hardware[0])
        self.s_n_gpus = tk.StringVar(value="1")
        self.bool_early_stop = tk.BooleanVar(value=True)
        self.s_early_stop_patience = tk.StringVar(value="10")
        self.s_batch_size = tk.StringVar(value="32")
        self.s_epochs = tk.StringVar(value="500")
        self.bool_shuffle = tk.BooleanVar(value=True)
        self.s_val_split = tk.StringVar(value="0.2")

        ##############################################################################
        # Variables for monitors submenu
        ##############################################################################
        self.bool_mse_monitor = tk.BooleanVar(value=False)
        self.bool_mae_monitor = tk.BooleanVar(value=False)
        self.bool_acc_monitor = tk.BooleanVar(value=True)

        ##############################################################################
        # Variables for save configurations submenu
        ##############################################################################
        self.bool_save_model = tk.BooleanVar(value=False)
        self.s_save_model_path = tk.StringVar()
        self.bool_save_csv = tk.BooleanVar(value=False)
        self.s_save_csv_path = tk.StringVar()
        self.bool_save_checkpoints = tk.BooleanVar(value=False)
        self.s_save_checkpoints_path = tk.StringVar()
        self.s_save_checkpoints_frequency = tk.StringVar(value="1")
        self.bool_tensorboard = tk.BooleanVar(value=False)
        self.s_tensorboard_path = tk.StringVar()
        self.s_tensorboard_frequency = tk.StringVar(value="5")

        ##############################################################################
        # Variables for BBD options
        ##############################################################################
        self.o_scaling = ('global', 'per predictor layer')
        self.s_scaling = tk.StringVar(value=self.o_scaling[0])
        self.s_scales = tk.StringVar(value="0.1, 0.9")
        self.o_aspect_ratios = ('global', 'per predictor layer')
        self.s_aspect_ratios = tk.StringVar(value=self.o_aspect_ratios[0])
        self.s_ARs = tk.StringVar(value="(0.5, 1.0, 1.5, 2.0)")
        self.s_n_classes = tk.StringVar(value="1")
        self.s_steps = tk.StringVar(value="(8, 16, 32, 64, 128)")
        self.s_offsets = tk.StringVar(value="None")
        self.s_variances = tk.StringVar(value="(0.1, 0.1, 0.2, 0.2)")
        self.s_conf_thresh = tk.StringVar(value="0.5")
        self.s_iou_thresh = tk.StringVar(value="0.5")
        self.s_top_k = tk.StringVar(value="200")
        self.s_nms_max_output = tk.StringVar(value="400")
        self.o_coords_type = ('centroids', 'minmax', 'corners')
        self.s_coords_type = tk.StringVar(value=self.o_coords_type[0])
        self.bool_2_for_1 = tk.BooleanVar(value=False)
        self.bool_clip_boxes = tk.BooleanVar(value=False)
        self.bool_norm_coords = tk.BooleanVar(value=False)
        self.s_pos_iou_thresh = tk.StringVar(value="0.5")
        self.s_neg_iou_limit = tk.StringVar(value="0.3")
