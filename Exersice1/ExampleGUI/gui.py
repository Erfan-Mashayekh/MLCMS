import sys
import tkinter
import json
from tkinter import Button, Canvas, Menu
from scenario import Scenario
from pedestrian import Pedestrian


class MainGUI():
    """
    Defines a simple graphical user interface.
    To start, use the `start_gui` method.
    """

    def create_scenario(self, ):
        print('create not implemented yet')


    def restart_scenario(self, ):
        print('restart not implemented yet')


    def step_scenario(self, scenario : Scenario,
                            canvas : Canvas,
                            canvas_image) -> None:
        """
        Moves the simulation forward by one step, and visualizes the result.

        Args:
            scenario (scenario_elements.Scenario): Add _description_
            canvas (tkinter.Canvas): Add _description_
            canvas_image (missing _type_): Add _description_
        """
        scenario.update_step()
        scenario.to_image(canvas, canvas_image)

    def load_scenario(self, path) -> Scenario:
        """
        Load a scenario that is specified in a JSON file
        TODO: missing obstacles read in
        Args:
            path: path to JSON file of scenario

        Returns:
            Scenario: the scenario specified by the JSON file
        """
        with open(path, 'r') as f:
            scenario_dict = json.load(f)
        x, y = scenario_dict['shape']
        sc = Scenario(x, y)
        for pos in scenario_dict['targets']:
            x, y = pos
            sc.grid[x, y] = Scenario.NAME2ID['TARGET']
        sc.recompute_target_distances()
        for pedestrian in scenario_dict['pedestrians']:
            pos, speed = pedestrian
            x, y = pos
            sc.pedestrians.append(Pedestrian((x, y), speed))
        return sc

    def exit_gui(self, ):
        """
        Close the GUI.
        """
        sys.exit()


    def start_gui(self, ):
        """
        Creates and shows a simple user interface with a menu and multiple buttons.
        Only one button works at the moment: "step simulation".
        Also creates a rudimentary, fixed Scenario instance with
        three Pedestrian instances and multiple targets.
        """
        win = tkinter.Tk()
        win.geometry('500x500')  # setting the size of the window
        win.title('Cellular Automata GUI')

        menu = Menu(win)
        win.config(menu=menu)
        file_menu = Menu(menu)
        menu.add_cascade(label='Simulation', menu=file_menu)
        file_menu.add_command(label='New', command=self.create_scenario)
        file_menu.add_command(label='Restart', command=self.restart_scenario)
        file_menu.add_command(label='Close', command=self.exit_gui)

        canvas = Canvas(win, width=Scenario.GRID_SIZE[0], height=Scenario.GRID_SIZE[1])  # creating the canvas
        canvas_image = canvas.create_image(5, 50, image=None, anchor=tkinter.NW)
        canvas.pack()

        # sc = self._create_default_scenario()
        sc = self.load_scenario("/home/ese/Courses/MLCMS/MLCMS/Exersice1/scenarios/sc0.json")

        # can be used to show pedestrians and targets
        sc.to_image(canvas, canvas_image)

        # can be used to show the target grid instead
        # sc.target_grid_to_image(canvas, canvas_image)

        btn = Button(win, text='Step simulation', command=lambda: self.step_scenario(sc, canvas, canvas_image))
        btn.place(x=20, y=10)
        btn = Button(win, text='Restart simulation', command=self.restart_scenario)
        btn.place(x=200, y=10)
        btn = Button(win, text='Create simulation', command=self.create_scenario)
        btn.place(x=380, y=10)

        win.mainloop()

    # private:
    def _create_default_scenario(self, ) -> Scenario:
        """
        Creates a fixed default Scenario instance with
        three Pedestrian instances and multiple targets.

        Returns:
            Scenario: the default scenario
        """
        sc = Scenario(100, 100)

        sc.grid[23, 25] = Scenario.NAME2ID['TARGET']
        sc.grid[23, 45] = Scenario.NAME2ID['TARGET']
        sc.grid[43, 55] = Scenario.NAME2ID['TARGET']
        sc.recompute_target_distances()

        sc.pedestrians = [
            Pedestrian((31, 2), 2.3),
            Pedestrian((1, 10), 2.1),
            Pedestrian((80, 70), 2.1)
        ]
        return sc