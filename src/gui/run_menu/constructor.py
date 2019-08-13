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
"""src/gui/run_menu/constructor.py"""


import tkinter as tk
from src.engine.constructor import Dlae
from src.utils.engine_utils import level_one_error_checking
from src.utils.engine_utils import level_two_error_checking


class RunMenuConstructor(tk.Menu):
    def __init__(self, parent, controller):
        tk.Menu.__init__(self, parent)
        self.parent = parent
        self.controller = controller
        self.run_menu = tk.Menu(self.parent)
        self.parent.add_cascade(label='Run', menu=self.run_menu)
        self.run_menu.add_command(label='Run engine', command=self.run_engine)

    def run_engine(self):
        configs = self.controller.get_configs()
        configs_lvl1, errors_lvl1, warnings_lvl1 = level_one_error_checking(configs)

        if any(warnings_lvl1):
            print('Level 1 warnings encountered.')
            print("The following level 1 warnings were identified and corrected based on engine defaults:")
            for warning in warnings_lvl1:
                print(warning)

        if any(errors_lvl1):
            print('Level 1 errors encountered.')
            print("Please fix the level 1 errors before continuing (Help menu > Errors).")
            [self.controller.errors_list_box.insert(tk.END, error) for error in errors_lvl1]
        else:
            configs_lvl2, errors_lvl2, warnings_lvl2 = level_two_error_checking(configs_lvl1)

            if any(warnings_lvl2):
                print('Level 2 warnings encountered.')
                print("The following level 2 warnings were identified and corrected based on engine defaults:")
                for warning in warnings_lvl2:
                    print(warning)

            if any(errors_lvl2):
                print('Level 2 errors encountered.')
                print("Please fix the level 2 errors before continuing (Help menu > Errors).")
                [self.controller.errors_list_box.insert(tk.END, error) for error in errors_lvl2]
            else:
                engine = Dlae(configs)
                if any(engine.errors):
                    print('Level 3 errors encountered.')
                    print("Please fix the level 3 errors before continuing (Help menu > Errors).")
                    [self.controller.errors_list_box.insert(tk.END, error) for error in engine.errors]
                else:
                    engine.run()
                    if any(engine.errors):
                        print('Level 3 errors encountered.')
                        print("Please fix the level 3 errors before continuing (Help menu > Errors).")
                        [self.controller.errors_list_box.insert(tk.END, error) for error in engine.errors]
