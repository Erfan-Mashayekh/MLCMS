
from megaman.geometry import Geometry
from scipy.sparse.csgraph import connected_components

def generate_geometry(radius):
    """
    Generates the k nearesr graph based on the cut-off radius around each datapoint
    """
    adjacency_method = 'cyflann'
    cyflann_kwds = {'index_type':'kmeans', 'branching':64, 'iterations':20, 'cb_index':0.4}
    adjacency_kwds = {'radius': radius}
    affinity_method = 'gaussian'
    affinity_kwds = {'radius':radius}
    laplacian_method = 'geometric'
    laplacian_kwds = {'scaling_epps':radius}

    return Geometry(adjacency_method=adjacency_method, adjacency_kwds=adjacency_kwds,
                    affinity_method=affinity_method, affinity_kwds=affinity_kwds,
                    laplacian_method=laplacian_method, laplacian_kwds=laplacian_kwds)


def check_connected_components(radius, geom):
    """
    Check if the selecet raduis generates the correct number of components
    In this case the value must be 1 since there is only one manifold (swiss roll)
    compute an adjacency matrix with a radius    
    """

    geom.adjacency_kwds = {'radius':radius}
    adjacency_matrix = geom.compute_adjacency_matrix()
    # compute the corresponding affinity matrix
    geom.affinity_kwds = {'radius':radius}
    affinity_matrix = geom.compute_affinity_matrix({'radius':radius})
    (number_connected_components, labels) = connected_components(affinity_matrix)
    print("The number of connected components is: ")
    print(number_connected_components)