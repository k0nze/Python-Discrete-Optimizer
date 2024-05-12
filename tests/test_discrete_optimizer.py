import unittest

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

        ps = ParameterSet([p0, p1, p2])
        ds = ps.get_design_space()

        self.assertIn([0, 0, 0], ds)
        self.assertIn([0, 1, 0], ds)
        self.assertIn([0, 2, 0], ds)
        self.assertIn([0, 3, 0], ds)

        self.assertIn([1, 0, 0], ds)
        self.assertIn([1, 1, 0], ds)
        self.assertIn([1, 2, 0], ds)
        self.assertIn([1, 3, 0], ds)

        self.assertIn([2, 0, 0], ds)
        self.assertIn([2, 1, 0], ds)
        self.assertIn([2, 2, 0], ds)
        self.assertIn([2, 3, 0], ds)

        self.assertIn([3, 2, 0], ds)
        self.assertIn([3, 3, 0], ds)
        self.assertIn([3, 4, 0], ds)
        self.assertIn([3, 5, 1], ds)

        self.assertIn([4, 2, 0], ds)
        self.assertIn([4, 3, 0], ds)
        self.assertIn([4, 4, 0], ds)
        self.assertIn([4, 5, 1], ds)

        self.assertIn([5, 2, 0], ds)
        self.assertIn([5, 3, 0], ds)
        self.assertIn([5, 4, 0], ds)
        self.assertIn([5, 5, 1], ds)

        self.assertEqual(len(ds), 24)

    def test_global_search(self):
        ...
