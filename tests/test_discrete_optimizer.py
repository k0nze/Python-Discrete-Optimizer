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

        for dp in ds:
            print(dp)
