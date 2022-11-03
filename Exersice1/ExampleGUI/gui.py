import sys
import os
import tkinter
import json
from tkinter import Button, Canvas, Menu, filedialog
from scenario import Scenario
from pedestrian import Pedestrian
from time import sleep


class MainGUI():
    """
    Defines a simple graphical user interface.
    To start, use the `start_gui` method.
    """
    # find path where this file is in
    _PATH = os.path.dirname(os.path.realpath(__file__))


    def restart_scenario(self, ):
        """restarts the scenario

        Raises:
            ValueError: if self.restart_dict contains invalid values
        """
        match self.restart_dict['type']:
            case 'load':
                self.load_scenario(*self.restart_dict['args'])
            case 'create':
                # TODO:
                print('create not implemented yet')
            case _ :
                raise ValueError('Could not match type of restart scenario')


    def step_scenario(self, canvas : Canvas,
                            canvas_image) -> None:
        """
        Moves the simulation forward by one step, and visualizes the result.
        """
        self.scenario.update_step()
        self.scenario.to_image(canvas, canvas_image)

    def play_scenario(self, canvas : Canvas,
                            canvas_image,
                            win : tkinter.Tk) -> None:
        """
        Moves the simulation forward by 50 steps, and visualizes the result.
        """
        for i in range(50):
            self.scenario.update_step()
            self.scenario.to_image(canvas, canvas_image)
            win.update()
            sleep(0.05)

    def toggle_distance_mode(self):
        self.scenario.toggle_distance_mode()
        self.scenario.recompute_target_distances()

    def load_scenario(self, canvas : Canvas,
                            canvas_image,
                            path : str = None):
        """
        Load a scenario that is specified in a JSON file

        Args:
            path: path to JSON file of scenario

        Returns:
            Scenario: the scenario specified by the JSON file
        """
        if path == None:
            initialdir = os.path.abspath(
                                os.path.join(self._PATH, "../scenarios")
                            )
            path = filedialog.askopenfilename(initialdir=initialdir)
        with open(path, 'r') as f:
            scenario_dict = json.load(f)
        x, y = scenario_dict['shape']
        sc = Scenario(x, y)
        for pos in scenario_dict['targets']:
            x, y = pos
            sc.grid[x, y] = Scenario.NAME2ID['TARGET']
            sc.targets.append([x, y])
        for pos in scenario_dict['obstacles']:
            x, y = pos
            sc.grid[x, y] = Scenario.NAME2ID['OBSTACLE']
        sc.recompute_target_distances()
        for pedestrian in scenario_dict['pedestrians']:
            pos, speed = pedestrian
            x, y = pos
            sc.pedestrians.append(Pedestrian((x, y), speed))
        self.scenario = sc
        self.scenario.to_image(canvas, canvas_image)
        self.restart_dict = {'type': 'load',
                             'args': (canvas, canvas_image, path)}


    def exit_gui(self, ):
        """
        Close the GUI.
        """
        sys.exit()

    def show_cost(self, canvas, canvas_image):
        self.scenario.grid_to_image('cost', canvas, canvas_image)

    def start_gui(self, ):
        """
        Creates and shows a simple user interface with a menu and multiple buttons.
        Only one button works at the moment: "step simulation".
        Also creates a rudimentary, fixed Scenario instance with
        three Pedestrian instances and multiple targets.
        """
        win = tkinter.Tk()
        win.geometry('1050x1050')  # setting the size of the window
        win.title('Cellular Automata GUI')

        canvas = Canvas(win, width=Scenario.GRID_SIZE[0],
                        height=Scenario.GRID_SIZE[1])  # creating the canvas
        canvas_image = canvas.create_image(5, 50, image=None, anchor=tkinter.NW)
        canvas.pack()

        menu = Menu(win)
        win.config(menu=menu)
        file_menu = Menu(menu)
        menu.add_cascade(label='Simulation', menu=file_menu)
        file_menu.add_command(label='Load Scenario',
                              command=lambda: self.load_scenario(canvas, canvas_image))
        file_menu.add_command(label='Restart', command=self.restart_scenario)
        file_menu.add_command(label='Close', command=self.exit_gui)


        path = os.path.abspath(
                        os.path.join(self._PATH, "../scenarios/default.json")
                    )
        self.load_scenario(canvas, canvas_image, path)

        btn = Button(win, text='Step',
                     command=lambda: self.step_scenario(canvas, canvas_image))
        btn.place(x=20, y=10)
        btn = Button(win, text='Play 50 Steps',
                     command=lambda: self.play_scenario(canvas, canvas_image, win))
        btn.place(x=100, y=10)
        btn = Button(win, text='Toggle Pedestrian Avoidance On/Off',
                     command=Pedestrian.toggle_pedestrian_avoidance)
        btn.place(x= 250, y=10)
        btn = Button(win, text='Restart',
                     command=self.restart_scenario)
        btn.place(x=600, y=10)
        btn = Button(win, text='Toggle Distance Mode',
                     command=self.toggle_distance_mode)
        btn.place(x=700, y=10)
        btn = Button(win, text='Show Cost',
                     command=lambda: self.show_cost(canvas, canvas_image))
        btn.place(x=900, y=10)

        win.mainloop()

