import unittest

from numpy.linalg import matrix_rank
from scipy import sparse

from sparse_null import sparse_null


class TestSparseNull(unittest.TestCase):

    def test_(self):
        for i in range(100):
            with self.subTest(i=i):
                A = sparse.random_array((150, 150), density=0.01, rng=i)
                H = sparse_null(A)
                self.assertTrue((A @ H).max() < 1e-14)

    def test_completeness(self):
        for i in range(100):
            with self.subTest(i=i):
                A = sparse.random_array((150, 150), density=0.01, rng=i)
                H = sparse_null(A)
                self.assertEqual(
                    int(matrix_rank(A.todense()) + matrix_rank(H.todense())), A.shape[1]
                )


if __name__ == "__main__":
    unittest.main()
