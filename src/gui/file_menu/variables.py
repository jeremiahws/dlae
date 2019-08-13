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
"""src/gui/file_menu/variables.py"""


import tkinter as tk


class FileMenuVariables(object):
    def __init__(self):
        self.load_file_path = ""
        self.load_ckpt_file_path = ""
        self.load_model_file_path = ""
        self.s_load_file_path = tk.StringVar(value=self.load_file_path)
        self.s_load_ckpt_file_path = tk.StringVar(value=self.load_ckpt_file_path)
        self.s_load_model_file_path = tk.StringVar(value=self.load_model_file_path)
        self.save_file_path = ""
