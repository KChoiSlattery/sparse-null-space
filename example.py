from scipy import sparse
from sparse_null import sparse_null

A = sparse.random_array((1000, 15000), density=0.001, rng=12)
H = sparse_null(A, show=True)

# Verify that each column of H produces 0 when multiplied into A
print((A @ H).max())