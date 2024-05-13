import unittest
from discrete_optimizer.discrete_optimizer import GlobalSearch, SimulatedAnnealing
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
        min_x, results, steps = gs.minimize(verbose=False)

        self.assertEqual(min_x[0], np.argmin(A))
        self.assertEqual(object_function(min_x), np.min(A))

    def test_global_search_2d(self):
        def fill(A):
            offset_x = A.shape[1] // 2
            offset_y = A.shape[0] // 2

            for x in range(A.shape[1]):
                for y in range(A.shape[0]):
                    A[x][y] = (x - offset_x) ** 2 + (y - offset_y) ** 2

        A_rows = 20
        A_cols = 20

        A = np.zeros(shape=(A_rows, A_cols))

        fill(A)

        object_function = lambda xs: A[xs[0]][xs[1]]

        x = Parameter("x", list(range(0, A_cols)))
        y = Parameter("y", list(range(0, A_rows)))

        ps = ParameterSet(x, y)

        gs = GlobalSearch(ps, object_function)
        min_x, results, steps = gs.minimize(verbose=False)

        self.assertEqual(object_function(min_x), np.min(A))

    def test_euclidean_distance(self):
        p = (1, 1, 1, 1)
        q = (1, 1, 1, 1)

        d = SimulatedAnnealing.euclidean_distance(p, q)
        self.assertEqual(d, 0.0)

        p = (1, 1, 1, 1)
        q = (1, 1, 1, 2)

        d = SimulatedAnnealing.euclidean_distance(p, q)
        self.assertGreater(d, 0.0)

    def test_pertubation_function(self):
        x = Parameter("x", list(range(0, 10)))
        y = Parameter("x", list(range(0, 20)))
        z = Parameter("x", list(range(0, 30)))

        ps = ParameterSet(
            x,
            y,
            z,
            exclude=[(3, 10, 28), (2, 11, 29), (4, 11, 29), (3, 10, 29), (3, 11, 28)],
        )

        design_space = ps.get_design_space()
        np_design_space = SimulatedAnnealing.convert_design_space_to_numpy(design_space)

        design_point = np.array((3, 11, 29))

        next_design_point = SimulatedAnnealing.pertubate(
            design_point, np_design_space, 1.1
        )
        np.testing.assert_equal(next_design_point, np.array((3, 12, 29)))

    def test_simulated_annealing_1d(self):
        def fill(A):
            offset = A.size // 2
            for x in range(A.size):
                A[x] = (x - offset) ** 2

        A_size = 20

        A = np.zeros(shape=(A_size,))

        fill(A)

        x = Parameter("x", list(range(0, A_size)))
        ps = ParameterSet(x)

        object_function = lambda xs: A[xs[0]]

        sa = SimulatedAnnealing(ps, object_function)
        min_x, results, steps = sa.minimize(verbose=False)

        self.assertLessEqual(object_function(min_x), np.min(A) + 1)

    def test_simulated_annealing_2d(self):
        def fill(A):
            offset_x = A.shape[1] // 2
            offset_y = A.shape[0] // 2

            for x in range(A.shape[1]):
                for y in range(A.shape[0]):
                    A[x][y] = (x - offset_x) ** 2 + (y - offset_y) ** 2

        A_rows = 40
        A_cols = 40

        A = np.zeros(shape=(A_rows, A_cols))

        fill(A)

        object_function = lambda xs: A[xs[0]][xs[1]]

        x = Parameter("x", list(range(0, A_cols)))
        y = Parameter("y", list(range(0, A_rows)))

        ps = ParameterSet(x, y)

        object_function = lambda xs: A[xs[0]][xs[1]]

        sa = SimulatedAnnealing(ps, object_function)
        min_x, results, steps = sa.minimize(verbose=False)

        self.assertLessEqual(object_function(min_x), np.min(A) + 1)


if __name__ == "__main__":
    unittest.main()
