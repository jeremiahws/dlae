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
"""dlae/gui/view_menu/architecture.py"""


import tkinter as tk


class Architecture:
    def __init__(self):
        self.tl_architecture = tk.Toplevel()
        self.tl_architecture.title('Network architecture')
        self.tl_architecture.wm_protocol('WM_DELETE_WINDOW', self.tl_architecture.withdraw)
        self.cv_architecture = tk.Canvas(self.tl_architecture, scrollregion=(0, 0, 10000, 10000))
        self.cv_architecture.config(width="30c", height="15c")
        self.sb_architecture_scroll_bar = tk.Scrollbar(self.tl_architecture, orient=tk.HORIZONTAL)
        self.sb_architecture_scroll_bar.pack(side=tk.BOTTOM, fill=tk.X)
        self.sb_architecture_scroll_bar.config(command=self.cv_architecture.xview)
        self.cv_architecture.config(xscrollcommand=self.sb_architecture_scroll_bar.set)
        self.cv_architecture.pack(fill=tk.BOTH, expand=True)
        self.tl_architecture.withdraw()

    def show(self):
        self.tl_architecture.deiconify()
