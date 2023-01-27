
import numpy as np
import scipy as sp

import datafold.dynfold as dfold
import datafold.pcfold as pfold

class DiffusionMap:
    
    def compute_eigenfunctions(self, positions, n_evecs, radius):
        """
        Compute the eigenfunctions of the Laplace-Beltrami operator for the, 
        given dataset 'positions'.
        Returns: eigenfunctions = [eigenvectors, eigenvalues]
        """

        # Compute distance 'd' using kd_tree algorithm
        kd_tree = sp.spatial.KDTree(positions)
        d = kd_tree.sparse_distance_matrix(kd_tree, radius).toarray()
        eps = 0.05*np.linalg.norm(kd_tree.maxes - kd_tree.mins)

        # Compute the kernel and normalize it
        w = np.exp(-d**2 / eps)
        w = np.where(w < 1, w, 0) # avoid zeros on non-diagonals get 1
        p = np.diag(np.sum(w,axis=1))
        p_inv = np.diag(p)**(-2) * p
        kernel = p_inv @ w @ p_inv

        # Compute eigendecompositions of T_hat
        q = np.diag(np.sum(kernel,axis=1))
        q_inv = np.diag(q)**(-2) * q
        t_hat = np.sqrt(q_inv) @ kernel @ np.sqrt(q_inv)
        t_hat_evals, t_hat_evecs = np.linalg.eigh(t_hat)

        # Get 'L+1' largest eigenvalues and eigenvectors of T_hat
        al = np.flip(t_hat_evals[-(n_evecs+1):], axis=-1)
        vl = np.flip(t_hat_evecs[:,-(n_evecs+1):], axis=-1)

        evals = al**(1/eps)  # TODO:square root check
        evecs = np.sqrt(q_inv) @ vl

        return evecs, evals

    def compute_eigenfunctions_datafold(self, positions, n_evecs):
        """
        Compute the eigenfunctions for the given dataset 'positions' using a Datafold.
        Returns: eigenfunctions = [eigenvectors, eigenvalues]
        """
        
        # Create the Manifold using Datafold lib
        positions_pcm = pfold.PCManifold(positions)
        positions_pcm.optimize_parameters()

        # Compute the kernel and the eigenfunctions using Datafold lib
        dmap = dfold.DiffusionMaps(
            kernel=pfold.GaussianKernel(epsilon=positions_pcm.kernel.epsilon),
            n_eigenpairs=n_evecs,
            dist_kwargs=dict(cut_off=positions_pcm.cut_off),
        )
        dmap = dmap.fit(positions_pcm)
        evecs, evals = dmap.eigenvectors_, dmap.eigenvalues_

        return evecs, evals


