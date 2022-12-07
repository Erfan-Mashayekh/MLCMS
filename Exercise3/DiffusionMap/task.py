
import numpy as np
import matplotlib.pyplot as plt

from diffusion_map import DiffusionMap
from datafold.utils.plot import plot_pairwise_eigenvector


class Task:
    
    def __init__(self, part, dataset, nr_samples, nr_samples_plot, sample, radius):
        self.part = part
        self.dataset = dataset
        self.nr_samples = nr_samples  
        self.nr_samples_plot = nr_samples_plot 
        self.sample = sample
        self.radius = radius

    def part1(self):
        """
        Compute the parametes required in part 1
        """
        positions, time = self.dataset.generate_periodic_data()
        dmap = DiffusionMap()
        evecs, evals = dmap.compute_eigenfunctions(positions, self.sample, self.radius)

        # Plot the eigenfunctions with respect to time
        self.dataset.plot_periodic_data(positions, time)
        self.plot_eigenfunctions(evecs, evals, time)

    def part2(self):
        """
        Compute the parametes required in part 2
        """
        positions, positions_color = self.dataset.generate_swiss_roll()
        dmap = DiffusionMap()
        if self.part == 2:
            evecs, evals = dmap.compute_eigenfunctions(positions, self.sample, self.radius)
        elif self.part == 4:
            evecs, evals = dmap.compute_eigenfunctions_datafold(positions, self.sample)

        # Plot the eigenfunctions with respect to time
        self.dataset.plot_swiss_roll(positions, positions_color)
        self.plot_eigenfunctions(evecs, evals, positions_color)
    
    def part3(self):
        """
        Compute the parametes required in part 3
        """
        positions = np.loadtxt("data_DMAP_PCA_vadere.txt", dtype=float)
        dmap = DiffusionMap()
        evecs, evals = dmap.compute_eigenfunctions(positions, self.sample, self.radius)
        #evecs, evals = dmap.compute_eigenfunctions_datafold(positions, self.sample)        

        fig = plt.figure()
        ax = plt.axes()
        ax.plot(positions[:,0], positions[:,1]);   # plot pedestrian 1
        ax.plot(positions[:,2], positions[:,3]);   # plot pedestrian 2
        fig.suptitle(f"Trajectory of the first two pedestrians")
        
        self.plot_eigenfunctions(evecs, evals, None)


    def plot_eigenfunctions(self, evecs, evals, time):
        """
        Plot different graphs with respect to selected part.
        """
        print(f"{self.sample} Largest eigenvalues with respect this dataset: \n {evals}")
        idx_plot = np.random.permutation(self.nr_samples)[0:self.nr_samples_plot]

        if self.part == 1:
            fig = plt.figure(figsize=(15, 7))
            plot = 0
            for evec in evecs.T:    
                plot += 1 
                ax  = fig.add_subplot(int(str(int(self.sample / 3 + 1)) +"3" + str(plot)))
                ax.scatter(time[idx_plot], evec[idx_plot], c=time[idx_plot])
                ax.set_xlabel("time")
                ax.set_ylabel(f"eigenvector {plot}")
            fig.suptitle(f"Eigenfunctions with respect to time, {self.nr_samples_plot} from {self.nr_samples} points.")
        elif self.part == 2 or self.part == 4:
            plot_pairwise_eigenvector(
                eigenvectors=evecs[idx_plot, :],
                n=1,
                fig_params=dict(figsize=[15, 7]),
                scatter_params=dict(cmap=plt.cm.Spectral, c=time[idx_plot]),
            ) 
        elif self.part == 3:
            plot_pairwise_eigenvector(
                eigenvectors=evecs,
                n=1,
                fig_params=dict(figsize=[15, 7])
            )
        plt.show()


