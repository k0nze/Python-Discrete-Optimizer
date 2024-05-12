import unittest
from discrete_optimizer.discrete_optimizer import GlobalSearch
import numpy as np

from discrete_optimizer import ListValues, Parameter, ParameterSet


class TestDiscreteOptimizer(unittest.TestCase):
    def test_list_values(self):
        ...

    def test_design_space_contraining(self):
        p0_values = ListValues([0, 1, 2, 3, 4, 5])
        p0 = Parameter("p0", {(None, None): p0_values})

        p1_values0 = ListValues([0, 1, 2, 3])
        p1_values1 = ListValues([2, 3, 4, 5])

        p1 = Parameter(
            "p1",
            {
                (p0, ListValues([0, 1, 2])): p1_values0,
                (p0, ListValues([3, 4, 5])): p1_values1,
            },
        )

        p2_values0 = ListValues([0])
        p2_values1 = ListValues([1])
        p2 = Parameter(
            "p2",
            {
                (p1, ListValues([0, 1, 2, 3, 4])): p2_values0,
                (p1, ListValues([5])): p2_values1,
            },
        )

        ps = ParameterSet(p0, p1, p2, exclude=[(0, 0, 0)])
        ds = ps.get_design_space()

        self.assertIn((0, 1, 0), ds)
        self.assertIn((0, 2, 0), ds)
        self.assertIn((0, 3, 0), ds)

        self.assertIn((1, 0, 0), ds)
        self.assertIn((1, 1, 0), ds)
        self.assertIn((1, 2, 0), ds)
        self.assertIn((1, 3, 0), ds)

        self.assertIn((2, 0, 0), ds)
        self.assertIn((2, 1, 0), ds)
        self.assertIn((2, 2, 0), ds)
        self.assertIn((2, 3, 0), ds)

        self.assertIn((3, 2, 0), ds)
        self.assertIn((3, 3, 0), ds)
        self.assertIn((3, 4, 0), ds)
        self.assertIn((3, 5, 1), ds)

        self.assertIn((4, 2, 0), ds)
        self.assertIn((4, 3, 0), ds)
        self.assertIn((4, 4, 0), ds)
        self.assertIn((4, 5, 1), ds)

        self.assertIn((5, 2, 0), ds)
        self.assertIn((5, 3, 0), ds)
        self.assertIn((5, 4, 0), ds)
        self.assertIn((5, 5, 1), ds)

        self.assertEqual(len(ds), 23)

    def test_global_search_1d(self):
        def fill(A):
            offset = A.size // 2
            for x in range(A.size):
                A[x] = (x - offset) ** 2

        A_size = 20
        A = np.zeros(shape=(A_size,))
        fill(A)

        object_function = lambda xs: A[xs[0]]

        x = Parameter("x", list(range(0, A_size)))
        ps = ParameterSet(x)

        gs = GlobalSearch(ps, object_function)
        min_x, results = gs.minimize(verbose=False)

        self.assertEqual(min_x[0], np.argmin(A))
        self.assertEqual(object_function(min_x), np.min(A))

    def test_global_search_2d(self):
        A_rows = 20
        A_cols = 20

        A = np.zeros(shape=(A_rows, A_cols))

        print(A)

        object_function = lambda xs: A[xs[0]][xs[1]]

        x = Parameter("x", list(range(0, A_cols)))
        y = Parameter("x", list(range(0, A_rows)))

        ps = ParameterSet(x, y)

        gs = GlobalSearch(ps, object_function)
        min_x, results = gs.minimize(verbose=True)

        # self.assertEqual(min_x[0], np.argmin(A))
        self.assertEqual(object_function(min_x), np.min(A))


if __name__ == "__main__":
    unittest.main()
