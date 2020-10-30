import unittest
import numpy as np

beta = 0.1
P = np.array([
[1 - np.exp(-4 * beta), np.exp(-4 * beta) / 3, np.exp(-4 * beta) / 3, np.exp(-4 * beta) / 3, 0, 0, 0, 0],
[1 / 3.0, 0, 0, 0, 1 / 3.0, 1 / 3.0, 0, 0],
[1 / 3.0, 0, 0, 0, 1 / 3.0, 0, 1 / 3.0, 0],
[1 / 3.0, 0, 0, 0, 0, 1 / 3.0, 1 / 3.0, 0],
[0, 1 / 3.0, 1 / 3.0, 0, 0, 0, 0, 1 / 3.0],
[0, 1 / 3.0, 0, 1 / 3.0, 0, 0, 0, 1 / 3.0],
[0, 0, 1 / 3.0, 1 / 3.0, 0, 0, 0, 1 / 3.0],
[0, 0, 0, 0, np.exp(-4 * beta) / 3, np.exp(-4 * beta) / 3, np.exp(-4 * beta) / 3, 1 - np.exp(-4 * beta)]
])
p1 = np.exp(3 * beta) / (2 * np.exp(3 * beta) + 6 * np.exp(-1 * beta))
p2 = np.exp(-1 * beta) / (2 * np.exp(3 * beta) + 6 * np.exp(-1 * beta))
class Markov(unittest.TestCase):
    def test_eigen(self):
        values, vectors = np.linalg.eig(P.T)
        for i in range(8):
            if abs(values[i] - 1) < 1e-5:
                l2norm = np.linalg.norm(vectors[:, i], ord=1)
                _pi = vectors[:, i] / l2norm
                if _pi[0] < 0:
                    _pi = -1.0 * _pi
                self.assertAlmostEqual(_pi[0], p1)
                self.assertAlmostEqual(_pi[1], p2)
                self.assertAlmostEqual(_pi[2], p2)
                self.assertAlmostEqual(_pi[3], p2)
                self.assertAlmostEqual(_pi[4], p2)
                self.assertAlmostEqual(_pi[5], p2)
                self.assertAlmostEqual(_pi[6], p2)
                self.assertAlmostEqual(_pi[7], p1)

if __name__ == '__main__':
    unittest.main()