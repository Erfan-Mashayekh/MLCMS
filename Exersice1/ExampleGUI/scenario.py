import numpy as np
from PIL import Image, ImageTk
import scipy.spatial.distance
import networkx as nx
from math import sqrt

from typing import List
from pedestrian import Pedestrian


class Scenario:
    """
    A scenario for a cellular automaton.
    """
    GRID_SIZE = (1000, 1000)
    ID2NAME = {
        0: 'EMPTY',
        1: 'TARGET',
        2: 'OBSTACLE',
        3: 'PEDESTRIAN'
    }
    NAME2COLOR = {
        'EMPTY': (0, 0, 0),
        'PEDESTRIAN': (255, 0, 0),
        'TARGET': (0, 0, 255),
        'OBSTACLE': (0, 255, 0),
        'PATH' : (150, 100, 100)
    }
    NAME2ID = {
        ID2NAME[0]: 0,
        ID2NAME[1]: 1,
        ID2NAME[2]: 2,
        ID2NAME[3]: 3
    }
    DELTA_T = 1.0

    # Hard-coded settings
    despawn = True      # If True pedestrians are removed from the scenario
                        #   once they reach a target
    distance_mode = 1   # 1 -> shortest path; 2 -> euclidean distance

    def __init__(self, width, height):
        if width < 1 or width > 1024:
            raise ValueError(f"Width {width} must be in [1, 1024].")
        if height < 1 or height > 1024:
            raise ValueError(f"Height {height} must be in [1, 1024].")

        self.width = width
        self.height = height
        self.grid_image = None
        self.grid = np.zeros((width, height))
        self.pedestrians : List[Pedestrian]
        self.pedestrians = []
        self.targets = []
        self.target_distance_grids = np.zeros((width, height))
        self.pedestrian_cost = np.zeros((width, height))
        self.cost = np.zeros((width, height))

    def recompute_target_distances(self):
        """
        computes the grid of distances to the nearest target.
        The path-finding algorithm that is used depends on self.distance_mode

        Raises:
            ValueError: if self.distance_mode is invalid
        """
        match self.distance_mode:
            case 0:
                self._compute_euclidean_distances()
            case 1:
                self._compute_walking_distances()
            case _:
                raise ValueError('invalid distance_mode')


    def _compute_walking_distances(self) -> None:
        """
        Computes walking distance, along "cell directions".
        The distance calculation account for obstacles.

        Writes the distance for every grid cell, as a np.ndarray to
        self.target_distance_grids
        """
        G = self._build_graph()
        sources = [j * self.width + i for [i, j] in self.targets]
        distance = nx.multi_source_dijkstra_path_length(G, sources)
        distance_grids = np.full((self.width, self.height), np.inf)

        for i in range(self.width):
            for j in range(self.height):
                if self._is_obstacle(i, j):
                    continue
                id = self.width * j + i # compute node id
                distance_grids[i,j] = distance[id]
        self.target_distance_grids = distance_grids


    def _is_obstacle(self, i : int, j : int) -> bool:
        """
        Convenience function to check grid position for obstacles.

        Args:
            i (int): horizontal position
            j (int): vertical position

        Returns:
            bool: true if there is an obstacle at the specified position
        """
        return self.grid[i, j] == self.NAME2ID['OBSTACLE']


    def _build_graph(self) -> nx.Graph:
        """
        builds graph used in dijkstra algorithm from given scenario grid

        Returns:
            nx.Graph: graph containing all non-obstacle cells
        """
        G = nx.Graph()
        G.add_nodes_from(range(self.width*self.height))

        obstacle_ids = []
        for i in range(self.width):
            for j in range(self.height):
                id = self.width * j + i # compute node id
                if self._is_obstacle(i, j):
                    obstacle_ids.append(id)
                    continue
                if i + 1 < self.width:
                    G.add_edge(id, id + 1, weight=1)
                    if j + 1 < self.height:
                        G.add_edge(id, id + 1 + self.width, weight=sqrt(2))
                if j + 1 < self.height:
                    G.add_edge(id, id + self.width, weight=1)
                    if i - 1 > -1:
                        G.add_edge(id, id - 1 + self.width, weight=sqrt(2))
        G.remove_nodes_from(obstacle_ids)
        return G


    def _compute_euclidean_distances(self) -> None:
        """
        Computes the shortest distance from every grid point to the nearest target cell.
        This does not take obstacles into account.

        Writes the distance for every grid cell, as a np.ndarray to
        self.target_distance_grids
        """
        targets = []
        for [x, y] in self.targets:
            targets.append([y, x]) # y and x are flipped because they are in image space.
        if len(targets) == 0:
            return np.zeros((self.width, self.height))

        targets = np.row_stack(targets)
        x_space = np.arange(0, self.width)
        y_space = np.arange(0, self.height)
        xx, yy = np.meshgrid(x_space, y_space)
        positions = np.column_stack([xx.ravel(), yy.ravel()])

        # after the target positions and all grid cell positions are stored,
        # compute the pair-wise distances in one step with scipy.
        distances = scipy.spatial.distance.cdist(targets, positions)

        # now, compute the minimum over all distances to all targets.
        distances = np.min(distances, axis=0)

        self.target_distance_grids = distances.reshape((self.width, self.height))

    def compute_overall_costs(self):
        self.compute_pedestrians_costs()
        self.cost = self.pedestrian_cost + self.target_distance_grids

    def compute_pedestrians_costs(self) -> None:
        """
        Computes the utility (cost) imposed on a pedestrian by other pedestrians.
        """
        self.pedestrian_cost = np.zeros((self.width, self.height))
        for pedestrian in self.pedestrians:
            pedestrian.add_compute_potential(self.pedestrian_cost)

    def update_step(self):
        """
        Updates the position of all pedestrians.
        This does not take obstacles or other pedestrians into account.
        Pedestrians can occupy the same cell.
        """
        self.compute_overall_costs()
        for pedestrian in self.pedestrians:
            pedestrian.update_step(self)

            # Despawn pedestrians once they reach a target
            x, y = pedestrian.position
            if self.despawn == True and [x, y] in self.targets:
                pedestrian.set_status_to_despawned()


    @staticmethod
    def cell_to_color(_id):
        return Scenario.NAME2COLOR[Scenario.ID2NAME[_id]]

    def grid_to_image(self, mode, canvas, old_image_id):
        """
        Creates a colored image based on the distance to the target stored in
        self.target_distance_gids.
        :param canvas: the canvas that holds the image.
        :param old_image_id: the id of the old grid image.
        """
        im = Image.new(mode="RGB", size=(self.width, self.height))
        pix = im.load()
        self.compute_overall_costs()
        for x in range(self.width):
            for y in range(self.height):
                if mode == 'cost':
                    grid = self.cost[x][y]
                elif mode == 'dist':
                    grid = self.target_distance_grids[x][y]
                else:
                    raise ValueError("Mode must either be 'cost' or 'dist'")
                if self._is_obstacle(x, y):
                    pix[x, y] = (255, 255, 255)
                    continue
                pix[x, y] = (max(0, min(255, int(10 * grid) - 0 * 255)),
                             max(0, min(255, int(10 * grid) - 1 * 255)),
                             max(0, min(255, int(10 * grid) - 2 * 255)))
        im = im.resize(Scenario.GRID_SIZE, Image.NONE)
        self.grid_image = ImageTk.PhotoImage(im)
        canvas.itemconfigure(old_image_id, image=self.grid_image)

    def to_image(self, canvas, old_image_id):
        """
        Creates a colored image based on the ids stored in self.grid.
        Pedestrians are drawn afterwards, separately.
        :param canvas: the canvas that holds the image.
        :param old_image_id: the id of the old grid image.
        """
        im = Image.new(mode="RGB", size=(self.width, self.height))
        pix = im.load()
        for x in range(self.width):
            for y in range(self.height):
                pix[x, y] = self.cell_to_color(self.grid[x, y])
        for pedestrian in self.pedestrians:
            for [x, y] in pedestrian.path:
                pix[x, y] = Scenario.NAME2COLOR['PATH']
            if pedestrian.status != 'despawned':
                x, y = pedestrian.position
                pix[x, y] = Scenario.NAME2COLOR['PEDESTRIAN']
        im = im.resize(Scenario.GRID_SIZE, Image.NONE)
        self.grid_image = ImageTk.PhotoImage(im)
        canvas.itemconfigure(old_image_id, image=self.grid_image)