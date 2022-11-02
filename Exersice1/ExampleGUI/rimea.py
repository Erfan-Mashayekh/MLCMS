import numpy as np
from scenario import Scenario
from pedestrian import Pedestrian
from tkinter import Canvas

def rimea_scenario_4(self, canvas : Canvas,
                            canvas_image,
                            path : str = None):
        """ Create environment for RiMEA scenario 4 in task 5 """
        sc = Scenario(3000, 3000)
        for x in range(0, 3000):
            y_1 = 1500
            y_2 = 1531
            sc.grid[x, y_1] = Scenario.NAME2ID['OBSTACLE']
            sc.grid[x, y_2] = Scenario.NAME2ID['OBSTACLE']
        sc.recompute_target_distances()

        for y in range(1501, 1530):
            sc.grid[2999, y] = Scenario.NAME2ID['TARGET']
            sc.targets.append([2999, y])

        possible_pedestrians = []

        for x in range(0, 427):
           for y in range(1501, 1530):
              possible_pedestrians.append((x, y))
        density_1 = np.random.choice(range(len(possible_pedestrians)), 6428)
        for id in density_1:
            x, y = possible_pedestrians[id]
            sc.pedestrians.append(Pedestrian((x, y), 1.2))

        for x in range(428, 857):
           for y in range(1501, 1530):
              possible_pedestrians.append((x, y))
        density_1 = np.random.choice(range(len(possible_pedestrians)), 12857)
        for id in density_1:
            x, y = possible_pedestrians[id]
            sc.pedestrians.append(Pedestrian((x, y), 1.2))

        for x in range(858, 1249):
           for y in range(1501, 1530):
              possible_pedestrians.append((x, y))
        density_1 = np.random.choice(range(len(possible_pedestrians)), 25714)
        for id in density_1:
            x, y = possible_pedestrians[id]
            sc.pedestrians.append(Pedestrian((x, y), 1.2))

        for x in range(1250, 1677):
           for y in range(1501, 1530):
              possible_pedestrians.append((x, y))
        density_1 = np.random.choice(range(len(possible_pedestrians)), 38571)
        for id in density_1:
            x, y = possible_pedestrians[id]
            sc.pedestrians.append(Pedestrian((x, y), 1.2))

        for x in range(1678, 2104):
           for y in range(1501, 1530):
              possible_pedestrians.append((x, y))
        density_1 = np.random.choice(range(len(possible_pedestrians)), 51428)
        for id in density_1:
            x, y = possible_pedestrians[id]
            sc.pedestrians.append(Pedestrian((x, y), 1.2))

        for x in range(2105, 2532):
           for y in range(1501, 1530):
              possible_pedestrians.append((x, y))
        density_1 = np.random.choice(range(len(possible_pedestrians)), 64285)
        for id in density_1:
            x, y = possible_pedestrians[id]
            sc.pedestrians.append(Pedestrian((x, y), 1.2))

        for x in range(2533, 2959):
           for y in range(1501, 1530):
              possible_pedestrians.append((x, y))
        density_1 = np.random.choice(range(len(possible_pedestrians)), 77142)
        for id in density_1:
            x, y = possible_pedestrians[id]
            sc.pedestrians.append(Pedestrian((x, y), 1.2))

        self.scenario = sc
        self.scenario.to_image(canvas, canvas_image)
        self.restart_dict = {'type': 'load',
                             'args': (canvas, canvas_image, path)}