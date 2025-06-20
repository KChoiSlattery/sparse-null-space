import numpy as np
from tqdm import tqdm
from scipy import sparse


def sparse_null(A, show=False, thresh=0.1):
    """Creates a sparse array whose columns span the null space of A. Python
    implementation of the algorithm from [1]. Directly copies the MATLAB implementation
    in [2], including many of the comments, but has some changes because of how sparse
    arrays are handled in Python.

    Written by Kieran Choi-Slattery, December 2024
    Based on MATLAB implementation by Martin Holters, 2013.

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
    H = sparse.eye_array(n).tocsc()

    At = A.T

    for i in tqdm(range(m), disable=not show):
        s = (At[:, [t[i]]].T @ H).reshape(H.shape[1])
        if s.count_nonzero() != 0:
            # Only consider non-zero entries in s
            jnz = np.array(s.nonzero()).ravel()

            # Filter based on the pivoting threshold
            j = jnz[np.abs(s.data) >= thresh * np.max(np.abs(s.data))]

            # From the permissible columns in H, find the one with the lowest number of
            # non-zero elements
            jj = np.argmin(H[:, j].count_nonzero(axis=0))
            j = j[jj]

            # Create s wih the j'th element removed
            sd = s.todense()
            s_new = sparse.hstack([sparse.csr_array(sd[:j]), sd[j + 1 :]])

            # The following matrix manipulation comes from [2]
            H = sparse.hstack([H[:, :j], H[:, j + 1 :]]) - H[:, [j]] @ s_new / sd[j]
    
    # Make sure that H has 1 or more columns
    if H.shape[1] == 0:
        raise ValueError("A must not be full rank")
    return H
