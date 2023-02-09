import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import distance
from scipy.sparse.linalg import lobpcg
import time

from sklearn.datasets import make_s_curve, make_swiss_roll

from sklearn.manifold import SpectralEmbedding as sk_le

from megaman.geometry import Geometry
from megaman.embedding import SpectralEmbedding as me_le
from megaman.embedding import Isomap

def plot_3d(X: np.ndarray, 
            Y: np.ndarray,
            title: str):
    '''
    plot 3d for swiss-roll
    '''
    x,y,z = X.T
    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes(projection='3d')
    col = ax.scatter3D(x, y, z, c=Y, s=50, alpha=0.8)
    ax.set_title(title)
    fig.colorbar(col, ax=ax, orientation="horizontal", shrink=0.6, aspect=60, pad=0.01)

def create_swiss_roll(n_samples: int):
    '''
    create swiss-roll dataset

    Args:
        n_samples(int): the number n of samples

    Returns:
        swiss_points(np.ndarray, shape (n, 3)): samples in swiss-roll
        swiss_color(np.ndarray, shape (n, 1)): the color of data points
    '''
    np.random.seed(42)
    swiss_points, swiss_color = make_swiss_roll(n_samples)
    plot_3d(swiss_points, swiss_color, "Original swiss roll samples")
    return swiss_points, swiss_color

def cal_dist(x: np.ndarray)->np.ndarray:
    '''
    calculate distance between data points in graph

    Args:
        x(np.ndarray, shape (n, m)): the matrix of data points, every data point is m-dimenstional 

    Returns:
        dist(np.ndarray, shape (n, n)): the matrix of the distances data points in graph
    '''
    dist = distance.cdist(x, x, 'euclidean')
    #returns the euclidean distance between any two points
    return dist

def rbf(dist: float,
        radius: float)->np.ndarray:
    '''
    rbf kernel function

    Args:
        dist(float): the matrix of distance
        radius(float): parameters in rbf

    Returns:
        np.ndarray, shape (n, n): the matrix of the distances between data poins
    '''
    return np.exp(-dist**2/ ((radius**2) ) )

def cal_W(data: np.ndarray,
          adjacency: str,
          radius: float,
          n_neighbors: int)->np.ndarray:
    '''
    calculate the weight matrix W

    Args:
        data(np.ndarray, shape (n, m)): the matrix of data points, every data point is m-dimenstional 
        adjacency(str): the method of constructi adjacency matrix
        radius(float): the radius to chooes neighbors
        n_neighbors(int): number of nearest neighbors for nearest neighbors graph

    Returns:
        W(np.ndarray, shape (n, n)): the weight matrix W for the graph
    '''
    dist = cal_dist(data)
    n = dist.shape[0]
    W = np.zeros((n, n))
    if adjacency == "knn":
        for i in range(n):
            index = np.argsort(dist[i])[1:1+n_neighbors]
            W[i, index] = rbf(dist[i, index], radius)
            W[index, i] = rbf(dist[index, i], radius)
    else:
        for i in range(n):
            index = np.where(dist[i]<radius)
            W[i, index] = rbf(dist[i, index], radius)
            W[index, i] = rbf(dist[index, i], radius)
    return W

def my_le(data: np.ndarray,
          n_dims: int,
          adjacency: str,
          radius: float,
          n_neighbors: int):
    '''
    do spectral embedding in my algorithm

    Args:
        data(np.ndarray, shape (n, m)): the matrix of data points, every data point is m-dimenstional 
        n_dim(int): he dimension of the projected subspace
        adjacency(str): the method of constructi adjacency matrix
        radius(float): the radius to chooes neighbors
        n_neighbors(int): number of nearest neighbors for nearest neighbors graph

    Returns:
        my_embed_spectral(np.ndarray, shape (n, n)): the matrix of the distances between data poins
        t1(float): the time to embedding
    '''
    t0 = time.time()
    N = data.shape[0]
    W = cal_W(data, adjacency, radius, n_neighbors)
    D = np.zeros_like(W)
    for i in range(N):
        D[i,i] = np.sum(W[i])

    L = D - W
    D_inv = np.linalg.inv(D)
    eig_val, eig_vec = np.linalg.eig(np.dot(D_inv, L))
    # Y = np.eye(D.shape[0])
    # eig_val, eig_vec = lobpcg(L, -D, largest=True)

    sort_index = np.argsort(eig_val)

    eig_val = eig_val[sort_index]
    # print("10 largest eigenvalues: ", eig_val[:10])

    sort_index = sort_index[0:n_dims+1]
    eig_vec_picked = eig_vec[:, sort_index]

    t1 = time.time() - t0
    print(t1)

    my_embed_spectral = eig_vec_picked[:,1:]
    return my_embed_spectral, t1

def sklearn_le(data: np.ndarray,
               n_dims: int,
               adjacency: str,
               radius: float,
               n_neighbors: int):
    '''
    do spectral embedding in sklearn

    Args:
        data(np.ndarray, shape (n, m)): the matrix of data points, every data point is m-dimenstional 
        n_dim(int): he dimension of the projected subspace
        adjacency(str): the method of constructi adjacency matrix
        radius(float): the radius to chooes neighbors
        n_neighbors(int): number of nearest neighbors for nearest neighbors graph

    Returns:
        sk_embed_spectral(np.ndarray, shape (n, n)): the matrix of the distances between data poins
        t1(float): the time to embedding
    '''
    if adjacency=="knn":
        embed_spectral = sk_le(n_components=n_dims, affinity='nearest_neighbors', n_neighbors=n_neighbors)
    else:
        embed_spectral = sk_le(n_components=n_dims, affinity='rbf', gamma=1/radius)
        # embed_spectral = sk_le(n_components=n_dims, n_neighbors=150)
    t0 = time.time()
    sk_embed_spectral = embed_spectral.fit_transform(data)
    t1 = time.time() - t0
    print(t1)
    return sk_embed_spectral, t1

def megaman_le(data: np.ndarray,
               n_dims: int,
               adjacency: str,
               radius: float,
               n_neighbors: int):
    '''
    do spectral embedding in megaman

    Args:
        data(np.ndarray, shape (n, m)): the matrix of data points, every data point is m-dimenstional 
        n_dim(int): he dimension of the projected subspace
        adjacency(str): the method of constructi adjacency matrix
        radius(float): the radius to chooes neighbors
        n_neighbors(int): number of nearest neighbors for nearest neighbors graph

    Returns:
        megaman_embed_spectral(np.ndarray, shape (n, n)): the matrix of the distances between data poins
        t1(float): the time to embedding
    '''
    if adjacency=="knn":
        n_neighbors = n_neighbors
        adjacency_method = 'cyflann'
        adjacency_kwds = {'n_neighbors':n_neighbors} # ignore distances above this radius
        affinity_method = 'gaussian'
        affinity_kwds = {'radius':radius} # W = exp(-||x - y||^2/radius^2) 
        laplacian_method = 'symmetricnormalized'
        laplacian_kwds = {'scaling_epps':radius} # scaling ensures convergence to Laplace-Beltrami operator

        geom = Geometry(adjacency_method=adjacency_method, adjacency_kwds=adjacency_kwds,
                        affinity_method=affinity_method, affinity_kwds=affinity_kwds,
                        laplacian_method=laplacian_method, laplacian_kwds=laplacian_kwds)
        
        megaman_spectral = me_le(n_components=n_dims, geom=geom)
    else:
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

        megaman_spectral = me_le(n_components=n_dims, eigen_solver='amg',geom=geom)
    t0 = time.time()
    megaman_embed_spectral = megaman_spectral.fit_transform(data)
    t1 = time.time() - t0
    print(t1)
    return megaman_embed_spectral, t1

def megaman_isomap(data: np.ndarray,
               n_dims: int,
               adjacency: str,
               radius: float,
               n_neighbors: int):
    '''
    do spectral embedding in megaman

    Args:
        data(np.ndarray, shape (n, m)): the matrix of data points, every data point is m-dimenstional 
        n_dim(int): he dimension of the projected subspace
        adjacency(str): the method of constructi adjacency matrix
        radius(float): the radius to chooes neighbors
        n_neighbors(int): number of nearest neighbors for nearest neighbors graph

    Returns:
        isomap_embed_spectral(np.ndarray, shape (n, n)): the matrix of the distances between data poins
        t1(float): the time to embedding
    '''
    if adjacency=="knn":
        n_neighbors = n_neighbors
        adjacency_method = 'cyflann'
        adjacency_kwds = {'n_neighbors':n_neighbors}
        affinity_method = 'gaussian'
        affinity_kwds = {'radius':radius}
        laplacian_method = 'symmetricnormalized'
        laplacian_kwds = {'scaling_epps':radius}

        geom = Geometry(adjacency_method=adjacency_method, adjacency_kwds=adjacency_kwds,
                        affinity_method=affinity_method, affinity_kwds=affinity_kwds,
                        laplacian_method=laplacian_method, laplacian_kwds=laplacian_kwds)
        
        isomap = Isomap(n_components=n_dims, eigen_solver='arpack', geom=geom)
    else:
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

        isomap = Isomap(n_components=n_dims, eigen_solver='arpack', geom=geom)
    t0 = time.time()
    isomap_embed_spectral = isomap.fit_transform(data)
    t1 = time.time() - t0
    print(t1)
    return isomap_embed_spectral, t1

def plot_results(embed_spectral: np.ndarray,
                 color: np.ndarray):
    '''
    plot the results
    '''
    n_dims = embed_spectral.shape[1]
    fig = plt.figure(figsize=(20,(n_dims)/2*10))
    for i in range(n_dims-2):
        ax = fig.add_subplot((n_dims)/2,2,i+1)
        ax.set_title('$\Phi 0 vs. \Phi'+str(i+1))
        ax.scatter(embed_spectral[:, 0], embed_spectral[:, i+1], s=0.5, c=color)

def test_swiss_roll(n_samples: int,
                    method: str,
                    n_dim: int,
                    adjacency: str,
                    radius=3.0,
                    n_neighbors=10,
                    plot=False):
    '''
    test spectral embedding in swiss-roll

    Args:
        n_samples(int): the number of samples,
        method(str): "my" -> use our algorithm; "sk" -> use se in sklearn; "megaman" -> use se in megaman; "isomap" -> use isomap in megaman
        n_dim(int): the dimension of the projected subspace
        adjacency(str): "knn" -> k-nearest-neighbors; "radius" -> radiusa-based
        radius(float): the radius to chooes neighbors
        plot(bool): True -> plot the results; False -> do not plot the results
        
    Returns:
        sk_embed_spectral(np.ndarray, shape (n, n)): the matrix of the distances between data poins
        t1(float): the time to embedding
    '''
    np.random.seed(42)
    points, color = make_swiss_roll(n_samples)
    if method == "my":
        embed_spectral, t1 = my_le(points, n_dim, adjacency, radius, n_neighbors)
    elif method == "sk":
        embed_spectral, t1 = sklearn_le(points, n_dim, adjacency, radius, n_neighbors)
    elif method == "megaman":
        embed_spectral, t1 = megaman_le(points, n_dim, adjacency, radius, n_neighbors)
    else: embed_spectral, t1 = megaman_isomap(points, n_dim, adjacency, radius, n_neighbors)
    if plot == True:
        plot_results(embed_spectral, color)
    return embed_spectral, t1, color