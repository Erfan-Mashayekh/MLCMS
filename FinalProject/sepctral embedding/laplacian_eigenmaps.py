import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import distance
import time

from sklearn.datasets import make_s_curve, make_swiss_roll

from sklearn.manifold import SpectralEmbedding as sk_le

from megaman.geometry import Geometry
from megaman.embedding import SpectralEmbedding as me_le

def plot_3d(X: np.ndarray, 
            Y: np.ndarray,
            title: str):
    '''
    plot 3d 
    '''
    x,y,z = X.T
    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes(projection='3d')
    col = ax.scatter3D(x, y, z, c=Y, s=50, alpha=0.8)
    ax.set_title(title)
    fig.colorbar(col, ax=ax, orientation="horizontal", shrink=0.6, aspect=60, pad=0.01)

def create_swiss_roll(n_samples):
    np.random.seed(42)
    swiss_points, swiss_color = make_swiss_roll(n_samples)
    plot_3d(swiss_points, swiss_color, "Original swiss roll samples")
    return swiss_points, swiss_color

def cal_dist(x):
    '''
    calcultae pairwise distance
    x: matrix
    dist: 
    '''
    dist = distance.cdist(x, x, 'euclidean')
    #returns the euclidean distance between any two points
    return dist

def rbf(dist, t):
    '''
    rbf kernel function


    '''
    return np.exp(- dist**2/ (2* (t**2) ) )

def cal_W(data: np.ndarray):
    '''
    calculate the weight matrix W


    '''
    radius = 1.5
    dist = cal_dist(data)
    dist[dist < radius] = 0
    W = rbf(dist, radius)

    return W

def my_le(data, n_dims):
    '''
    do spectral embedding through myself
    :param data: (n_samples, n_features)
    :param n_dims: target dim
    :param n_neighbors: k nearest neighbors
    :param t: a param for rbf
    :return:
    '''
    t0 = time.time()
    N = data.shape[0]
    W = cal_W(data)
    D = np.zeros_like(W)
    for i in range(N):
        D[i,i] = np.sum(W[i])

    D_inv = np.linalg.inv(D)
    L = D - W
    eig_val, eig_vec = np.linalg.eig(np.dot(D_inv, L))

    sort_index_ = np.argsort(eig_val)

    eig_val = eig_val[sort_index_]
    print("eig_val[:10]: ", eig_val[:10])

    sort_index_ = sort_index_[0:n_dims]
    eig_val_picked = eig_val[0:n_dims]
    print(eig_val_picked)
    eig_vec_picked = eig_vec[:, sort_index_]

    t1 = time.time() - t0
    print(t1)

    X_ndim = eig_vec_picked
    return X_ndim

def sklearn_le(data, n_dims):
    '''
    do spectral embedding through sklearn
    '''
    embed_spectral = sk_le(n_components=n_dims, affinity='rbf', gamma= 0.25, eigen_solver='amg')
    # embed_spectral = sk_le(n_components=n_dims, n_neighbors=150)
    t0 = time.time()
    sklearn_embed_spectral = embed_spectral.fit_transform(data)
    t1 = time.time() - t0
    print(t1)
    return sklearn_embed_spectral

def megaman_le(data, n_dims):
    '''
    do spectral embedding through megaman
    '''
    radius = 0.94
    # n_neighbors = 100
    # adjacency_kwds = {'n_neighbors':n_neighbors}
    adjacency_method = 'cyflann'
    # radius = 1.49/(data.shape[0])**(1.0/(2+6))
    adjacency_kwds = {'radius':radius}
    # cyflann_kwds = {'index_type':'kmeans', 'branching':64, 'iterations':20, 'cb_index':0.4}
    # adjacency_kwds = {'radius':radius, 'cyflann_kwds':cyflann_kwds}
    affinity_method = 'gaussian'
    affinity_kwds = {'radius':radius}
    laplacian_method = 'symmetricnormalized'
    laplacian_kwds = {'scaling_epps':radius}

    geom = Geometry(adjacency_method=adjacency_method, adjacency_kwds=adjacency_kwds,
                    affinity_method=affinity_method, affinity_kwds=affinity_kwds,
                    laplacian_method=laplacian_method, laplacian_kwds=laplacian_kwds)
    
                    
                    
    geom.set_data_matrix(data)

    megaman_spectral = me_le(n_components=n_dims, eigen_solver='amg',geom=geom, drop_first=False)
    t0 = time.time()
    megaman_embed_spectral = megaman_spectral.fit_transform(data)
    t1 = time.time() - t0
    print(t1)
    return megaman_embed_spectral

def plot_results(embed_spectral, color):
    '''
    plot the results
    '''
    n_dims = embed_spectral.shape[1]
    fig = plt.figure(figsize=(20,(n_dims-1)/2*10))
    for i in range(n_dims-2):
        ax = fig.add_subplot((n_dims-1)/2,2,i+1)
        # ax.set_title("$\Phi_{0} vs. \Phi"+{i+1})
        ax.scatter(embed_spectral[:, 0], embed_spectral[:, i+1], s=0.5, c=color)