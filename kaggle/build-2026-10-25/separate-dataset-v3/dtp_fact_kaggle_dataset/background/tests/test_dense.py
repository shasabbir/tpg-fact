import unittest

import numpy as np

from retrieval.dense import cosine_rank


class DenseRetrievalTest(unittest.TestCase):
    def test_cosine_rank(self):
        ranked = cosine_rank(
            np.array([1.0, 0.0]),
            np.array([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]]),
            ["orthogonal", "exact", "partial"],
        )
        self.assertEqual([item[0] for item in ranked], ["exact", "partial", "orthogonal"])


if __name__ == "__main__":
    unittest.main()
