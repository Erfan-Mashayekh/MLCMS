import numpy as np


class Pedestrian:
    """
    Defines a single pedestrian.
    """
    R_MAX = 5               # Radius used in avoidance potential
    INTERACTION_SCALE = 3.0 # Characteristic scale of avoidance potential
    status = 'walking'

    def __init__(self, position, desired_speed):
        self._position = position
        self._desired_speed = desired_speed
        self._path = []


    @property
    def position(self):
        return self._position

    @property
    def desired_speed(self):
        return self._desired_speed

    @property
    def path(self):
        return self._path

    def set_status_to_despawned(self):
        self.status = 'despawned'

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
        Moves to the cell with the lowest cost.
        Pedestrians can occupy the same cell, if the avoidance term is too low.

        :param scenario: The current scenario instance.
        """
        self._path.append(self._position)

        distance = scenario.target_distance_grids
        cost = scenario.cost
        self.add_compute_potential(cost, sign= -1) # remove self-interaction

        # Search for optimal step
        neighbors = self.get_neighbors(scenario)
        next_cell_cost = cost[self._position[0]][self._position[1]]
        x, y = self._position
        for (n_x, n_y) in neighbors:
            if next_cell_cost > cost[n_x, n_y]: # TODO: Think about better ways
                self._position = (n_x, n_y)
                next_cell_cost = cost[n_x, n_y]

    def add_compute_potential(self, cost_grid : np.ndarray,
                                    sign = +1):
        """
        adds the cost (negative utility) added by the pedestrian to the
        cost_grid.

        Args:
            cost_array (np.ndarray): cost used in scenario
            sign (int, optional): allows to remove self-interaction.
                                Defaults to +1.
        """
        if self.status == 'despawned':
            return
        width, height = cost_grid.shape
        r_max_sq = self.R_MAX**2
        for dx in range(-self.R_MAX, self.R_MAX+1):
            for dy in range(-self.R_MAX, self.R_MAX+1):
                r_sq = dx**2 + dy**2
                if r_sq < r_max_sq:
                    pos_x, pos_y = self._position
                    x , y = (pos_x + dx, pos_y + dy)
                    # check in bounds
                    if -1 < x and x < width and -1 < y and y < height:
                        cost_grid[x, y] += sign * self.INTERACTION_SCALE \
                                            * np.exp(1 / (r_sq - r_max_sq))

