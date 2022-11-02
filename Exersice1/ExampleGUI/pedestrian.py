import numpy as np


class Pedestrian:
    """
    Defines a single pedestrian.
    """

    def __init__(self, position, desired_speed):
        self._position = position
        self._desired_speed = desired_speed
        self._path = []
        self._traversed_distance = 0
        self._distance_error = -1
        self._step = 0


    @property
    def position(self):
        return self._position

    @property
    def desired_speed(self):
        return self._desired_speed

    @property
    def path(self):
        return self._path

    def get_neighbors(self, scenario):
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

    def update_step(self, scenario):
        """
        Moves to the cell with the lowest distance to the target.
        This does not take obstacles or other pedestrians into account.
        Pedestrians can occupy the same cell.

        :param scenario: The current scenario instance.
        """
        # check whehter the pedestrian is slower or faster that desired speed
        if self._distance_error <= -0.5*scenario.DELTA_X:
            while self._distance_error <= -0.5*scenario.DELTA_X:
                # Search for optimal step
                self._path.append(self._position)
                neighbors = self.get_neighbors(scenario)
                next_cell_distance = scenario.target_distance_grids[self._position[0]][self._position[1]]
                next_pos = self._position
                for (n_x, n_y) in neighbors:
                    if next_cell_distance > scenario.target_distance_grids[n_x, n_y]:
                        next_pos = (n_x, n_y)
                        next_cell_distance = scenario.target_distance_grids[n_x, n_y]

                desired_traversed_distance = self._desired_speed * self._step * scenario.DELTA_T
                self._traversed_distance = self._traversed_distance \
                    + np.sqrt((self._position[0]-next_pos[0])**2 + (self._position[1]-next_pos[1])**2) * scenario.DELTA_X
                
                self._distance_error = self._traversed_distance - desired_traversed_distance

                self._position = next_pos
                if next_cell_distance == 0:
                    break
        else:
            desired_traversed_distance = self._desired_speed * self._step * scenario.DELTA_T
            self._distance_error = self._traversed_distance - desired_traversed_distance
        self._step = self._step + 1
        
        
        
