
import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import make_s_curve, make_swiss_roll


class Dataset:

    def __init__(self, nr_samples, nr_samples_plot):
        self.nr_samples = nr_samples          # Number of samples
        self.nr_samples_plot = nr_samples_plot  # Number of samples to plot
    
    def generate_periodic_data(self):
        """
        Generate 2d point cloud for the periodic dataset 
        Task: part1
        """
        k = np.linspace(1, self.nr_samples, self.nr_samples)
        time = 2 * np.pi * k / (self.nr_samples + 1)
        positions = np.asarray([np.cos(time), np.sin(time)]).T 

        return positions, time

    def plot_periodic_data(self, positions, time):
        """
        Plot 2d point cloud for the periodic dataset with respect to time
        Task: part1
        """
        fig = plt.figure(figsize=(4, 4))
        ax  = fig.add_subplot()
        idx_plot = np.random.permutation(self.nr_samples)[0:self.nr_samples_plot]
        ax.scatter(positions[idx_plot, 0], positions[idx_plot, 1], c=time[idx_plot],);
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Point cloud on a round periodic data")
    
    def generate_swiss_roll(self):
        """
        Generate 3d Swiss-roll point cloud
        Task: part2
        """

        return make_swiss_roll(self.nr_samples)

    def plot_swiss_roll(self, positions, positions_color):
        """
        Plot 3d Swiss-roll point cloud
        Task: part2
        """
        fig = plt.figure(figsize=[7, 7])
        ax = fig.add_subplot(1, 1, 1, projection="3d")
        ax.scatter(*positions.T, c=positions_color, cmap=plt.cm.Spectral, s=0.5)
        ax.set_title("Swiss roll: sampled manifold point cloud")


def generate_word2vec_dataset(nr_samples):
    import gensim.downloader
    google_news_vecs = gensim.downloader.load("word2vec-google-news-300")
    vecs = google_news_vecs.vectors
    idx = np.random.permutation(vecs.shape[0])[0:nr_samples]

    return vecs, idx