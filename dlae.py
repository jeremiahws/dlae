#!/usr/bin/env python

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
"""dlae.py"""


import sys
from src.gui.constructor import DlaeGui
from src.engine.constructor import Dlae
from src.utils.engine_utils import level_one_error_checking
from src.utils.engine_utils import level_two_error_checking
from src.utils.general_utils import load_config
import tkinter as tk


if __name__ == '__main__':
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
        configs = load_config(config_file)
        configs_lvl1, errors_lvl1, warnings_lvl1 = level_one_error_checking(configs)

        if any(warnings_lvl1):
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
                if any(engine.errors):
                    print('Level 3 errors encountered.')
                    print("Please fix the level 3 errors below before continuing:")
                    for error in engine.errors:
                        print(error)
                else:
                    engine.run()
                    if any(engine.errors):
                        print('Level 3 errors encountered.')
                        print("Please fix the level 3 errors below before continuing:")
                        for error in engine.errors:
                            print(error)

    else:
        root = tk.Tk()
        root.resizable(width=False, height=False)
        root.tk.call('wm', 'iconphoto', root._w, tk.PhotoImage(file="src/gui/button_figs/other/icon.png"))
        DlaeGui(root).grid()
        try:
            root.mainloop()
        except KeyboardInterrupt:
            quit()
