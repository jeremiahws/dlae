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
"""src/utils/gui_utils.py"""


import tkinter as tk
from src.gui.file_menu.variables import FileMenuVariables
from src.gui.data_menu.variables import DataMenuVariables
from src.gui.layers_menu.variables import LayersMenuVariables
from src.gui.options_menu.variables import OptionsMenuVariables
from src.gui.home_menu.variables import HomeMenuVariables


class GuiParameterController(object):
    def __init__(self):
        self.file_menu = FileMenuVariables()
        self.data_menu = DataMenuVariables()
        self.layers_menu = LayersMenuVariables()
        self.options_menu = OptionsMenuVariables()
        self.tools_menu = None
        self.home_menu = HomeMenuVariables()
        self.layers_list_box = None
        self.layers_list_box_serial = None
        self.layers_list_box_gen = None
        self.layers_list_box_discrim = None
        self.errors_list_box = None

    def get_configs(self):
        configs = {}
        configs['config_file'] = {}
        configs['config_file']['model_signal'] = self.home_menu.s_model_signal.get()
        configs['config_file']['type_signal'] = self.home_menu.s_type_signal.get()
        configs['config_file']['input_shape'] = self.home_menu.s_input_shape.get()
        configs['paths'] = {}
        configs['paths']['load_config'] = self.file_menu.s_load_file_path.get()
        configs['paths']['load_checkpoint'] = self.file_menu.s_load_ckpt_file_path.get()
        configs['paths']['load_model'] = self.file_menu.s_load_model_file_path.get()
        configs['paths']['train_X'] = self.data_menu.s_train_X_path.get()
        configs['paths']['train_y'] = self.data_menu.s_train_y_path.get()
        configs['paths']['validation_X'] = self.data_menu.s_val_X_path.get()
        configs['paths']['validation_y'] = self.data_menu.s_val_y_path.get()
        configs['paths']['test_X'] = self.data_menu.s_test_X_path.get()
        configs['preprocessing'] = {}
        configs['preprocessing']['minimum_image_intensity'] = self.data_menu.s_data_min.get()
        configs['preprocessing']['maximum_image_intensity'] = self.data_menu.s_data_max.get()
        configs['preprocessing']['image_context'] = self.data_menu.s_image_context.get()
        configs['preprocessing']['normalization_type'] = self.data_menu.s_normalization_type.get()
        configs['preprocessing']['categorical_switch'] = str(self.data_menu.bool_to_categorical.get())
        configs['preprocessing']['categories'] = self.data_menu.s_num_categories.get()
        configs['preprocessing']['weight_loss_switch'] = str(self.data_menu.bool_weight_loss.get())
        configs['preprocessing']['repeat_X_switch'] = str(self.data_menu.bool_repeatX.get())
        configs['preprocessing']['repeat_X_quantity'] = self.data_menu.s_repeatX.get()
        configs['augmentation'] = {}
        configs['augmentation']['apply_augmentation_switch'] = str(self.data_menu.bool_augmentation.get())
        configs['augmentation']['featurewise_centering_switch'] = str(self.data_menu.bool_fw_centering.get())
        configs['augmentation']['samplewise_centering_switch'] = str(self.data_menu.bool_sw_centering.get())
        configs['augmentation']['featurewise_normalization_switch'] = str(self.data_menu.bool_fw_normalization.get())
        configs['augmentation']['samplewise_normalization_switch'] = str(self.data_menu.bool_sw_normalization.get())
        configs['augmentation']['width_shift'] = self.data_menu.s_width_shift.get()
        configs['augmentation']['height_shift'] = self.data_menu.s_height_shift.get()
        configs['augmentation']['rotation_range'] = self.data_menu.s_rotation_range.get()
        configs['augmentation']['brightness_range'] = self.data_menu.s_brightness_range.get()
        configs['augmentation']['shear_range'] = self.data_menu.s_shear_range.get()
        configs['augmentation']['zoom_range'] = self.data_menu.s_zoom_range.get()
        configs['augmentation']['channel_shift_range'] = self.data_menu.s_channel_shift_range.get()
        configs['augmentation']['fill_mode'] = self.data_menu.s_fill_mode.get()
        configs['augmentation']['cval'] = self.data_menu.s_cval.get()
        configs['augmentation']['horizontal_flip_switch'] = str(self.data_menu.bool_horizontal_flip.get())
        configs['augmentation']['vertical_flip_switch'] = str(self.data_menu.bool_vertical_flip.get())
        configs['augmentation']['rounds'] = self.data_menu.s_rounds.get()
        configs['augmentation']['zca_epsilon'] = self.data_menu.s_zca_epsilon.get()
        configs['augmentation']['random_seed'] = self.data_menu.s_random_seed.get()
        configs['loss_function'] = {}
        configs['loss_function']['loss'] = self.options_menu.s_loss.get()
        configs['loss_function']['parameter1'] = self.options_menu.s_loss_param1.get()
        configs['loss_function']['parameter2'] = self.options_menu.s_loss_param2.get()
        configs['learning_rate_schedule'] = {}
        configs['learning_rate_schedule']['learning_rate'] = self.options_menu.s_base_lr.get()
        configs['learning_rate_schedule']['learning_rate_decay_factor'] = self.options_menu.s_lr_decay.get()
        configs['learning_rate_schedule']['decay_on_plateau_switch'] = str(self.options_menu.bool_decay_on_plateau.get())
        configs['learning_rate_schedule']['decay_on_plateau_factor'] = self.options_menu.s_decay_on_plateau_factor.get()
        configs['learning_rate_schedule']['decay_on_plateau_patience'] = self.options_menu.s_decay_on_plateau_patience.get()
        configs['learning_rate_schedule']['step_decay_switch'] = str(self.options_menu.bool_step_decay.get())
        configs['learning_rate_schedule']['step_decay_factor'] = self.options_menu.s_step_decay_factor.get()
        configs['learning_rate_schedule']['step_decay_period'] = self.options_menu.s_step_decay_period.get()
        configs['learning_rate_schedule']['discriminator_learning_rate'] = self.options_menu.s_d_lr.get()
        configs['learning_rate_schedule']['gan_learning_rate'] = self.options_menu.s_gan_lr.get()
        configs['optimizer'] = {}
        configs['optimizer']['optimizer'] = self.options_menu.s_optimizer.get()
        configs['optimizer']['beta1'] = self.options_menu.s_optimizer_beta1.get()
        configs['optimizer']['beta2'] = self.options_menu.s_optimizer_beta2.get()
        configs['optimizer']['rho'] = self.options_menu.s_optimizer_rho.get()
        configs['optimizer']['momentum'] = self.options_menu.s_optimizer_momentum.get()
        configs['optimizer']['epsilon'] = self.options_menu.s_optimizer_epsilon.get()
        configs['optimizer']['discriminator_optimizer'] = self.options_menu.s_d_optimizer.get()
        configs['optimizer']['gan_optimizer'] = self.options_menu.s_gan_optimizer.get()
        configs['training_configurations'] = {}
        configs['training_configurations']['hardware'] = self.options_menu.s_hardware.get()
        configs['training_configurations']['number_of_gpus'] = self.options_menu.s_n_gpus.get()
        configs['training_configurations']['early_stop_switch'] = str(self.options_menu.bool_early_stop.get())
        configs['training_configurations']['early_stop_patience'] = self.options_menu.s_early_stop_patience.get()
        configs['training_configurations']['batch_size'] = self.options_menu.s_batch_size.get()
        configs['training_configurations']['epochs'] = self.options_menu.s_epochs.get()
        configs['training_configurations']['shuffle_data_switch'] = str(self.options_menu.bool_shuffle.get())
        configs['training_configurations']['validation_split'] = self.options_menu.s_val_split.get()
        configs['monitors'] = {}
        configs['monitors']['mse_switch'] = str(self.options_menu.bool_mse_monitor.get())
        configs['monitors']['mae_switch'] = str(self.options_menu.bool_mae_monitor.get())
        configs['monitors']['accuracy_switch'] = str(self.options_menu.bool_acc_monitor.get())
        configs['save_configurations'] = {}
        configs['save_configurations']['save_model_switch'] = str(self.options_menu.bool_save_model.get())
        configs['save_configurations']['save_model_path'] = self.options_menu.s_save_model_path.get()
        configs['save_configurations']['save_csv_switch'] = str(self.options_menu.bool_save_csv.get())
        configs['save_configurations']['save_csv_path'] = self.options_menu.s_save_csv_path.get()
        configs['save_configurations']['save_checkpoints_switch'] = str(self.options_menu.bool_save_checkpoints.get())
        configs['save_configurations']['save_checkpoints_path'] = self.options_menu.s_save_checkpoints_path.get()
        configs['save_configurations']['save_checkpoints_frequency'] = self.options_menu.s_save_checkpoints_frequency.get()
        configs['save_configurations']['save_tensorboard_switch'] = str(self.options_menu.bool_tensorboard.get())
        configs['save_configurations']['save_tensorboard_path'] = self.options_menu.s_tensorboard_path.get()
        configs['save_configurations']['save_tensorboard_frequency'] = self.options_menu.s_tensorboard_frequency.get()
        configs['bbd_options'] = {}
        configs['bbd_options']['scaling_type'] = self.options_menu.s_scaling.get()
        configs['bbd_options']['scales'] = self.options_menu.s_scales.get()
        configs['bbd_options']['aspect_ratios_type'] = self.options_menu.s_aspect_ratios.get()
        configs['bbd_options']['aspect_ratios'] = self.options_menu.s_ARs.get()
        configs['bbd_options']['number_classes'] = self.options_menu.s_n_classes.get()
        configs['bbd_options']['steps'] = self.options_menu.s_steps.get()
        configs['bbd_options']['offsets'] = self.options_menu.s_offsets.get()
        configs['bbd_options']['variances'] = self.options_menu.s_variances.get()
        configs['bbd_options']['confidence_threshold'] = self.options_menu.s_conf_thresh.get()
        configs['bbd_options']['iou_threshold'] = self.options_menu.s_iou_thresh.get()
        configs['bbd_options']['top_k'] = self.options_menu.s_top_k.get()
        configs['bbd_options']['nms_maximum_output'] = self.options_menu.s_nms_max_output.get()
        configs['bbd_options']['coordinates_type'] = self.options_menu.s_coords_type.get()
        configs['bbd_options']['two_boxes_for_AR1_switch'] = str(self.options_menu.bool_2_for_1.get())
        configs['bbd_options']['clip_boxes_switch'] = str(self.options_menu.bool_clip_boxes.get())
        configs['bbd_options']['normalize_coordinates_switch'] = str(self.options_menu.bool_norm_coords.get())
        configs['bbd_options']['positive_iou_threshold'] = self.options_menu.s_pos_iou_thresh.get()
        configs['bbd_options']['negative_iou_limit'] = self.options_menu.s_neg_iou_limit.get()
        configs['layers'] = {}
        configs['layers']['serial_layer_list'] = self.layers_list_box_serial.get(0, tk.END)
        configs['layers']['generator_layer_list'] = self.layers_list_box_gen.get(0, tk.END)
        configs['layers']['discriminator_layer_list'] = self.layers_list_box_discrim.get(0, tk.END)

        return configs

    def set_configs(self, configs):
        self.home_menu.s_model_signal.set(configs['config_file']['model_signal'])
        self.home_menu.s_type_signal.set(configs['config_file']['type_signal'])
        self.home_menu.s_input_shape.set(configs['config_file']['input_shape'])
        self.file_menu.s_load_file_path.set(configs['paths']['load_config'])
        self.file_menu.s_load_ckpt_file_path.set(configs['paths']['load_checkpoint'])
        self.file_menu.s_load_model_file_path.set(configs['paths']['load_model'])
        self.data_menu.s_train_X_path.set(configs['paths']['train_X'])
        self.data_menu.s_train_y_path.set(configs['paths']['train_y'])
        self.data_menu.s_val_X_path.set(configs['paths']['validation_X'])
        self.data_menu.s_val_y_path.set(configs['paths']['validation_y'])
        self.data_menu.s_test_X_path.set(configs['paths']['test_X'])
        self.data_menu.s_data_min.set(configs['preprocessing']['minimum_image_intensity'])
        self.data_menu.s_data_max.set(configs['preprocessing']['maximum_image_intensity'])
        self.data_menu.s_image_context.set(configs['preprocessing']['image_context'])
        self.data_menu.s_normalization_type.set(configs['preprocessing']['normalization_type'])
        self.data_menu.bool_to_categorical.set(configs['preprocessing']['categorical_switch'])
        self.data_menu.s_num_categories.set(configs['preprocessing']['categories'])
        self.data_menu.bool_weight_loss.set(configs['preprocessing']['weight_loss_switch'])
        self.data_menu.bool_repeatX.set(configs['preprocessing']['repeat_X_switch'])
        self.data_menu.s_repeatX.set(configs['preprocessing']['repeat_X_quantity'])
        self.data_menu.bool_augmentation.set(configs['augmentation']['apply_augmentation_switch'])
        self.data_menu.bool_fw_centering.set(configs['augmentation']['featurewise_centering_switch'])
        self.data_menu.bool_sw_centering.set(configs['augmentation']['samplewise_centering_switch'])
        self.data_menu.bool_fw_normalization.set(configs['augmentation']['featurewise_normalization_switch'])
        self.data_menu.bool_sw_normalization.set(configs['augmentation']['samplewise_normalization_switch'])
        self.data_menu.s_width_shift.set(configs['augmentation']['width_shift'])
        self.data_menu.s_height_shift.set(configs['augmentation']['height_shift'])
        self.data_menu.s_rotation_range.set(configs['augmentation']['rotation_range'])
        self.data_menu.s_brightness_range.set(configs['augmentation']['brightness_range'])
        self.data_menu.s_shear_range.set(configs['augmentation']['shear_range'])
        self.data_menu.s_zoom_range.set(configs['augmentation']['zoom_range'])
        self.data_menu.s_channel_shift_range.set(configs['augmentation']['channel_shift_range'])
        self.data_menu.s_fill_mode.set(configs['augmentation']['fill_mode'])
        self.data_menu.s_cval.set(configs['augmentation']['cval'])
        self.data_menu.bool_horizontal_flip.set(configs['augmentation']['horizontal_flip_switch'])
        self.data_menu.bool_vertical_flip.set(configs['augmentation']['vertical_flip_switch'])
        self.data_menu.s_rounds.set(configs['augmentation']['rounds'])
        self.data_menu.s_zca_epsilon.set(configs['augmentation']['zca_epsilon'])
        self.data_menu.s_random_seed.set(configs['augmentation']['random_seed'])
        self.options_menu.s_loss.set(configs['loss_function']['loss'])
        self.options_menu.s_loss_param1.set(configs['loss_function']['parameter1'])
        self.options_menu.s_loss_param2.set(configs['loss_function']['parameter2'])
        self.options_menu.s_base_lr.set(configs['learning_rate_schedule']['learning_rate'])
        self.options_menu.s_lr_decay.set(configs['learning_rate_schedule']['learning_rate_decay_factor'])
        self.options_menu.bool_decay_on_plateau.set(configs['learning_rate_schedule']['decay_on_plateau_switch'])
        self.options_menu.s_decay_on_plateau_factor.set(configs['learning_rate_schedule']['decay_on_plateau_factor'])
        self.options_menu.s_decay_on_plateau_patience.set(configs['learning_rate_schedule']['decay_on_plateau_patience'])
        self.options_menu.bool_step_decay.set(configs['learning_rate_schedule']['step_decay_switch'])
        self.options_menu.s_step_decay_factor.set(configs['learning_rate_schedule']['step_decay_factor'])
        self.options_menu.s_step_decay_period.set(configs['learning_rate_schedule']['step_decay_period'])
        self.options_menu.s_d_lr.set(configs['learning_rate_schedule']['discriminator_learning_rate'])
        self.options_menu.s_gan_lr.set(configs['learning_rate_schedule']['gan_learning_rate'])
        self.options_menu.s_optimizer.set(configs['optimizer']['optimizer'])
        self.options_menu.s_optimizer_beta1.set(configs['optimizer']['beta1'])
        self.options_menu.s_optimizer_beta2.set(configs['optimizer']['beta2'])
        self.options_menu.s_optimizer_rho.set(configs['optimizer']['rho'])
        self.options_menu.s_optimizer_momentum.set(configs['optimizer']['momentum'])
        self.options_menu.s_optimizer_epsilon.set(configs['optimizer']['epsilon'])
        self.options_menu.s_d_optimizer.set(configs['optimizer']['discriminator_optimizer'])
        self.options_menu.s_gan_optimizer.set(configs['optimizer']['gan_optimizer'])
        self.options_menu.s_hardware.set(configs['training_configurations']['hardware'])
        self.options_menu.s_n_gpus.set(configs['training_configurations']['number_of_gpus'])
        self.options_menu.bool_early_stop.set(configs['training_configurations']['early_stop_switch'])
        self.options_menu.s_early_stop_patience.set(configs['training_configurations']['early_stop_patience'])
        self.options_menu.s_batch_size.set(configs['training_configurations']['batch_size'])
        self.options_menu.s_epochs.set(configs['training_configurations']['epochs'])
        self.options_menu.bool_shuffle.set(configs['training_configurations']['shuffle_data_switch'])
        self.options_menu.s_val_split.set(configs['training_configurations']['validation_split'])
        self.options_menu.bool_mse_monitor.set(configs['monitors']['mse_switch'])
        self.options_menu.bool_mae_monitor.set(configs['monitors']['mae_switch'])
        self.options_menu.bool_acc_monitor.set(configs['monitors']['accuracy_switch'])
        self.options_menu.bool_save_model.set(configs['save_configurations']['save_model_switch'])
        self.options_menu.s_save_model_path.set(configs['save_configurations']['save_model_path'])
        self.options_menu.bool_save_csv.set(configs['save_configurations']['save_csv_switch'])
        self.options_menu.s_save_csv_path.set(configs['save_configurations']['save_csv_path'])
        self.options_menu.bool_save_checkpoints.set(configs['save_configurations']['save_checkpoints_switch'])
        self.options_menu.s_save_checkpoints_path.set(configs['save_configurations']['save_checkpoints_path'])
        self.options_menu.s_save_checkpoints_frequency.set(configs['save_configurations']['save_checkpoints_frequency'])
        self.options_menu.bool_tensorboard.set(configs['save_configurations']['save_tensorboard_switch'])
        self.options_menu.s_tensorboard_path.set(configs['save_configurations']['save_tensorboard_path'])
        self.options_menu.s_tensorboard_frequency.set(configs['save_configurations']['save_tensorboard_frequency'])
        self.options_menu.s_scaling.set(configs['bbd_options']['scaling_type'])
        self.options_menu.s_scales.set(configs['bbd_options']['scales'])
        self.options_menu.s_aspect_ratios.set(configs['bbd_options']['aspect_ratios_type'])
        self.options_menu.s_ARs.set(configs['bbd_options']['aspect_ratios'])
        self.options_menu.s_n_classes.set(configs['bbd_options']['number_classes'])
        self.options_menu.s_steps.set(configs['bbd_options']['steps'])
        self.options_menu.s_offsets.set(configs['bbd_options']['offsets'])
        self.options_menu.s_variances.set(configs['bbd_options']['variances'])
        self.options_menu.s_conf_thresh.set(configs['bbd_options']['confidence_threshold'])
        self.options_menu.s_iou_thresh.set(configs['bbd_options']['iou_threshold'])
        self.options_menu.s_top_k.set(configs['bbd_options']['top_k'])
        self.options_menu.s_nms_max_output.set(configs['bbd_options']['nms_maximum_output'])
        self.options_menu.s_coords_type.set(configs['bbd_options']['coordinates_type'])
        self.options_menu.bool_2_for_1.set(configs['bbd_options']['two_boxes_for_AR1_switch'])
        self.options_menu.bool_clip_boxes.set(configs['bbd_options']['clip_boxes_switch'])
        self.options_menu.bool_norm_coords.set(configs['bbd_options']['normalize_coordinates_switch'])
        self.options_menu.s_pos_iou_thresh.set(configs['bbd_options']['positive_iou_threshold'])
        self.options_menu.s_neg_iou_limit.set(configs['bbd_options']['negative_iou_limit'])
        self.layers_list_box.delete(0, tk.END)
        self.layers_list_box_serial.delete(0, tk.END)
        self.layers_list_box_gen.delete(0, tk.END)
        self.layers_list_box_discrim.delete(0, tk.END)
        [self.layers_list_box_serial.insert(tk.END, layer) for layer in configs['layers']['serial_layer_list']]
        if any(configs['layers']['generator_layer_list']):
            [self.layers_list_box.insert(tk.END, layer) for layer in configs['layers']['generator_layer_list']]
        else:
            [self.layers_list_box.insert(tk.END, layer) for layer in configs['layers']['serial_layer_list']]
        [self.layers_list_box_gen.insert(tk.END, layer) for layer in configs['layers']['generator_layer_list']]
        [self.layers_list_box_discrim.insert(tk.END, layer) for layer in configs['layers']['discriminator_layer_list']]

        if any(configs['layers']['serial_layer_list']):
            self.home_menu.s_model_built.set('Serial model built')
        elif any(configs['layers']['generator_layer_list']) and any(configs['layers']['discriminator_layer_list']):
            self.home_menu.s_model_built.set('Gen & discrim built')
        elif not any(configs['layers']['serial_layer_list']):
            self.home_menu.s_model_built.set('No layers defined')
        else:
            self.home_menu.s_model_built.set('Multiple models defined')

        return
