import numpy as np

def pca(data):
    """
    Parameters
    ----------
    data: np.ndarray, shape [N, D]
        dataset be do SVD
    Returns
    -------
    U: numpy.ndarray, shape [N, N]
        u matrix
    S: numpy.ndarray, shape [N, D]
        sigma matrix
    Vh: numpy.ndarray,  shape [D,D]
        transpose of v matrix
    energy: numpy.ndarray,  shape [D,1]
        the energy matrix
    """
    #center the data
    mean = data.mean(axis=0)
    data_centered = data - mean

    #decompose the data through SVD
    U, s, Vh = np.linalg.svd(data_centered)

    # construct the S matrix
    s_square = np.square(s)
    trace = np.sum(s_square)
    S = np.zeros(shape=(data_centered.shape[0], data_centered.shape[1]))
    s = np.diag(s)
    S[:s.shape[0], :s.shape[1]] = s
    energy = s_square/trace

    return U, S, Vh, energy