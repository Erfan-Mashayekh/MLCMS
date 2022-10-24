import numpy as np
from scenario import Scenario


class Pedestrian:
    """
    Defines a single pedestrian.
    """

    def __init__(self, position, desired_speed):
        self._position = position
        self._desired_speed = desired_speed

    @property
    def position(self):
        return self._position

    @property
    def desired_speed(self):
        return self._desired_speed

    def get_neighbors(self, scenario : Scenario):
        """
        Compute all neighbors in a 9 cell neighborhood of the current position.
        :param scenario: The scenario instance.
        :return: A list of neighbor cell indices (x,y) around the current position.
        """
        return [
            (int(x + self._position[0]), int(y + self._position[1]))
            for x in [-1, 0, 1]
            for y in [-1, 0, 1]
            if 0 <= x + self._position[0] < scenario.width \
                    and 0 <= y + self._position[1] < scenario.height \
                    and np.abs(x) + np.abs(y) > 0
        ]

    def update_step(self, scenario : Scenario):
        """
        Moves to the cell with the lowest distance to the target.
        This does not take obstacles or other pedestrians into account.
        Pedestrians can occupy the same cell.

        :param scenario: The current scenario instance.
        """
        neighbors = self.get_neighbors(scenario)
        next_cell_distance = scenario.target_distance_grids[self._position[0]][self._position[1]]
        next_pos = self._position
        # Search for optimal step
        for (n_x, n_y) in neighbors:
            if next_cell_distance > scenario.target_distance_grids[n_x, n_y]:
                next_pos = (n_x, n_y)
                next_cell_distance = scenario.target_distance_grids[n_x, n_y]
        self._position = next_pos