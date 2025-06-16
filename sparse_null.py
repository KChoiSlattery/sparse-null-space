import numpy as np
from tqdm import tqdm
from scipy import sparse


def sparse_null(A, show=False, thresh=0.1):
    """Creates a sparse array whose columns span the null space of A. Python
    implementation of the algorithm from [1]. Directly copies the MATLAB implementation
    in [2], including most of the comments, but has some changes because of how sparse
    arrays are handled in Python.

    Kieran Choi-Slattery, December 2024

    [1] M. Khorramizadeh and N. Mahdavi-Amiri, "An efficient algorithm for sparse null
    space basis problem using ABS methods," Numerical Algorithms, vol. 62, no. 3, pp.
    469–485, Jun. 2012.

    [2] Martin Holters (2024). Null space for sparse matrix
    (https://www.mathworks.com/matlabcentral/fileexchange/42922-null-space-for-sparse-matrix),
    MATLAB Central File Exchange. Retrieved December 19, 2024.

    Args:
        A (array_like): The array for which a sparse basis of its null space is desired.
        thresh (float, optional): A parameter that focuses either on sparsity (thresh=0)
            or accuracy (thresh=1). Defaults to 0.1.
        show (bool, optional): Whether to display a progress bar. Defaults to False.

    Returns:
        csc_array: A sparse array in CSC format whose columns span the null space of A.
    """

    m, n = A.shape
    
    A = sparse.csr_array(A)
    
    # Get ordering of the rows of A by ascending count of non-zero elements
    r = np.diff(A.indptr)
    t = np.argsort(r)
    
    # Initialize H1
    H = sparse.eye(n).tocsc()
    
    At = A.T
    
    for i in range(m):
        s = (At[:, [t[i]]].T @ H).reshape(H.shape[1])
        if len(s.data) != 0:
            # Only consider non-zero entries in s
            jnz=s.nonzero()[1]
            
            # Filter based on the pivoting threshold
            jnz_filter_inds = np.argwhere(np.abs(s.data) >= thresh*np.max(np.abs(s))).ravel()
            j = jnz[jnz_filter_inds]
            
            # From the permissible columns in H, find the one with the lowest number of non-zero elements
            nonzero_in_cols = np.asarray(H[:, j].astype(bool).sum(axis=0))[0]
            jj = np.argmin(nonzero_in_cols)
            j = j[jj]
            
            
            
            sd = s.todense()
            H = sparse.hstack([H[:, :j-1], H[:, j:]])-H[:,j] @ sparse.hstack([sd[:j-1], sd[j:]])/sd[j]
    return H