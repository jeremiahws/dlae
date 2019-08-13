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
"""src/gui/options_menu/bbd_options.py"""


import tkinter as tk


class BbdOptions:
    def __init__(self, controller):
        self.controller = controller
        self.button_heights = 1
        self.button_widths = 15
        self.label_heights = 1
        self.label_widths = 15
        self.entry_widths = 15

        self.tl_bbd_options = tk.Toplevel()
        self.tl_bbd_options.title('BBD options')
        self.tl_bbd_options.resizable(width=False, height=False)
        self.tl_bbd_options.wm_protocol('WM_DELETE_WINDOW', self.tl_bbd_options.withdraw)

        self.l_scaling = tk.Label(self.tl_bbd_options, text='Scaling:').grid(row=0, column=0, sticky=tk.N+tk.S+tk.E+tk.W)
        self.om_scaling = tk.OptionMenu(self.tl_bbd_options, self.controller.options_menu.s_scaling, *self.controller.options_menu.o_scaling)
        self.om_scaling.config()
        self.om_scaling.grid(row=0, column=1, sticky=tk.N+tk.S+tk.E+tk.W)
        self.e_scaling = tk.Entry(self.tl_bbd_options, textvariable=self.controller.options_menu.s_scales).grid(row=0, column=2, sticky=tk.N+tk.S+tk.E+tk.W)

        self.l_aspect_ratios = tk.Label(self.tl_bbd_options, text='Aspect ratios:').grid(row=1, column=0, sticky=tk.N+tk.S+tk.E+tk.W)
        self.om_aspect_ratios = tk.OptionMenu(self.tl_bbd_options, self.controller.options_menu.s_aspect_ratios, *self.controller.options_menu.o_aspect_ratios)
        self.om_aspect_ratios.config()
        self.om_aspect_ratios.grid(row=1, column=1, sticky=tk.N + tk.S + tk.E + tk.W)
        self.e_aspect_ratios = tk.Entry(self.tl_bbd_options, textvariable=self.controller.options_menu.s_ARs).grid(row=1, column=2, sticky=tk.N+tk.S+tk.E+tk.W)

        self.l_n_classes = tk.Label(self.tl_bbd_options, text='Number of classes:').grid(row=2, column=0, sticky=tk.N+tk.S+tk.E+tk.W)
        self.e_n_classes = tk.Entry(self.tl_bbd_options, textvariable=self.controller.options_menu.s_n_classes).grid(row=2, column=1, sticky=tk.N+tk.S+tk.E+tk.W)

        self.l_steps = tk.Label(self.tl_bbd_options, text='Steps:').grid(row=3, column=0, sticky=tk.N+tk.S+tk.E+tk.W)
        self.e_steps = tk.Entry(self.tl_bbd_options, textvariable=self.controller.options_menu.s_steps).grid(row=3, column=1, sticky=tk.N+tk.S+tk.E+tk.W)

        self.l_offsets = tk.Label(self.tl_bbd_options, text='Offsets:').grid(row=4, column=0, sticky=tk.N+tk.S+tk.E+tk.W)
        self.e_offsets = tk.Entry(self.tl_bbd_options, textvariable=self.controller.options_menu.s_offsets).grid(row=4, column=1, sticky=tk.N+tk.S+tk.E+tk.W)

        self.l_variances = tk.Label(self.tl_bbd_options, text='Variances:').grid(row=5, column=0, sticky=tk.N+tk.S+tk.E+tk.W)
        self.e_variances = tk.Entry(self.tl_bbd_options, textvariable=self.controller.options_menu.s_variances).grid(row=5, column=1, sticky=tk.N+tk.S+tk.E+tk.W)

        self.l_conf_thresh = tk.Label(self.tl_bbd_options, text='Confidence threshold:').grid(row=6, column=0, sticky=tk.N+tk.S+tk.E+tk.W)
        self.e_conf_thresh = tk.Entry(self.tl_bbd_options, textvariable=self.controller.options_menu.s_conf_thresh).grid(row=6, column=1, sticky=tk.N+tk.S+tk.E+tk.W)

        self.l_iou_thresh = tk.Label(self.tl_bbd_options, text='IoU threshold:').grid(row=7, column=0, sticky=tk.N+tk.S+tk.E+tk.W)
        self.e_iou_thresh = tk.Entry(self.tl_bbd_options, textvariable=self.controller.options_menu.s_iou_thresh).grid(row=7, column=1, sticky=tk.N+tk.S+tk.E+tk.W)

        self.l_top_k = tk.Label(self.tl_bbd_options, text='Top k:').grid(row=8, column=0, sticky=tk.N+tk.S+tk.E+tk.W)
        self.e_top_k = tk.Entry(self.tl_bbd_options, textvariable=self.controller.options_menu.s_top_k).grid(row=8, column=1, sticky=tk.N+tk.S+tk.E+tk.W)

        self.l_nms_max_output = tk.Label(self.tl_bbd_options, text='NMS maximum output size:').grid(row=9, column=0, sticky=tk.N+tk.S+tk.E+tk.W)
        self.e_nms_max_output = tk.Entry(self.tl_bbd_options, textvariable=self.controller.options_menu.s_nms_max_output).grid(row=9, column=1, sticky=tk.N+tk.S+tk.E+tk.W)

        self.l_coords_type = tk.Label(self.tl_bbd_options, text='Coordinate type:').grid(row=10, column=0, sticky=tk.N+tk.S+tk.E+tk.W)
        self.om_coords_type = tk.OptionMenu(self.tl_bbd_options, self.controller.options_menu.s_coords_type, *self.controller.options_menu.o_coords_type)
        self.om_coords_type.config()
        self.om_coords_type.grid(row=10, column=1, sticky=tk.N+tk.S+tk.E+tk.W)

        self.l_2_for_1 = tk.Label(self.tl_bbd_options, text='Two boxes for AR=1:').grid(row=11, column=0, sticky=tk.N+tk.S+tk.E+tk.W)
        self.c_2_for_1 = tk.Checkbutton(self.tl_bbd_options, variable=self.controller.options_menu.bool_2_for_1).grid(row=11, column=1, sticky=tk.N+tk.S+tk.E+tk.W)

        self.l_clip_boxes = tk.Label(self.tl_bbd_options, text='Clip boxes:').grid(row=12, column=0, sticky=tk.N+tk.S+tk.E+tk.W)
        self.c_clip_boxes = tk.Checkbutton(self.tl_bbd_options, variable=self.controller.options_menu.bool_clip_boxes).grid(row=12, column=1, sticky=tk.N+tk.S+tk.E+tk.W)

        self.l_norm_coords = tk.Label(self.tl_bbd_options, text='Normalize coordinates:').grid(row=13, column=0, sticky=tk.N+tk.S+tk.E+tk.W)
        self.c_norm_coords = tk.Checkbutton(self.tl_bbd_options, variable=self.controller.options_menu.bool_norm_coords).grid(row=13, column=1, sticky=tk.N+tk.S+tk.E+tk.W)

        self.l_pos_iou_thresh = tk.Label(self.tl_bbd_options, text='Positive IoU threshold:').grid(row=14, column=0, sticky=tk.N+tk.S+tk.E+tk.W)
        self.e_pos_iou_thresh = tk.Entry(self.tl_bbd_options, textvariable=self.controller.options_menu.s_pos_iou_thresh).grid(row=14, column=1, sticky=tk.N+tk.S+tk.E+tk.W)
        self.l_neg_iou_limit = tk.Label(self.tl_bbd_options, text='Negative IoU limit:').grid(row=15, column=0, sticky=tk.N+tk.S+tk.E+tk.W)
        self.e_neg_iou_limit = tk.Entry(self.tl_bbd_options, textvariable=self.controller.options_menu.s_neg_iou_limit).grid(row=15, column=1, sticky=tk.N+tk.S+tk.E+tk.W)
        self.tl_bbd_options.withdraw()

    def show(self):
        self.tl_bbd_options.deiconify()
