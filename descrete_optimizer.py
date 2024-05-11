from typing import Iterable, List, Optional, Tuple, Dict
from copy import copy


class Values:
    def __init__(self) -> None:
        self.values = None

    def get_first_value(self):
        return None

    def __contains__(self, key) -> bool:
        return False

    def __iter__(self):
        return self 

    def __next__(self):
        raise StopIteration

    def __repr__(self) -> str:
        return str(self.values)

    def __hash__(self) -> int:
        if self.values is not None:
            return hash(None)
        else:
            return hash(tuple(self.values))


class RangeValues(Values):
    def __init__(self, value_range: Iterable[int]) -> None:
        super().__init__()
        self.values = value_range 

    def __contains__(self, key) -> bool:
        return (key in self.values) 

    def __iter__(self):
        self.iter_range = copy(self.values)
        return self
   
    def __next__(self):
        return next(self.iter_range) 

    def get_first_value(self):
        return list(self.values)[0]


class ListValues(Values):
    def __init__(self, values: List[int]) -> None:
        super().__init__()
        self.values = values 

    def __iter__(self):
        self.iter_index = 0
        return self
   
    def __next__(self):
        if self.iter_index < len(self.values):
            return_value = self.values[self.iter_index]
            self.iter_index += 1
            return return_value
        else:
            raise StopIteration


    def __contains__(self, key) -> bool:
        return (key in self.values)

    def get_first_value(self):
        return self.values[0]


class Parameter:
    def __init__(self, name: str, dependencies: Dict[Tuple[Optional["Parameter"], Optional[Values]], Values]) -> None:
        self.name = name
        # CAUTION: currently only one dependency is supported
        self.dependencies = dependencies
        self.current_value = None 
  
    def __repr__(self) -> str:
        return f"{self.name}={self.current_value}"

    def __hash__(self) -> int:
        return hash(self.name)


    def infer_initial_value(self):
        # if there is no dependency to another parameter set it to the first value for values
        if (None, None) in self.dependencies.keys():
            self.current_value = self.dependencies[(None, None)].get_first_value()
        else:
            # get current values of all parameter dependencies
            dependency_parameters = set()

            for parameter, value_range in self.dependencies.keys():
                dependency_parameters.add(parameter)

            # for each parameter infer the initial value
            for parameter in dependency_parameters:
                parameter.infer_initial_value()

            possible_values = self.constrain_values_via_dependencies()
            if len(possible_values) == 0:
                self.current_value = None
            else:
                self.current_value = possible_values[0]

    def constrain_values_via_dependencies(self):
        possible_values = set()

        for (constraining_parameter, contraining_value_range), possible_value_range in self.dependencies.items():
            # check if the current value of constraining parameter is in the constraining range
            # if this is the case add the values of the possible value range to values
            if constraining_parameter.current_value in contraining_value_range:
                for value in possible_value_range:
                    possible_values.add(value)

        possible_values = sorted(list(possible_values))
        return possible_values


class ParameterSet:
    def __init__(self, parameters: List[Parameter]) -> None:
        # CAUTION: parameters need to be ordered by their dependencies
        self.parameters = parameters
        # initialize parameters
        for parameter in parameters:
            parameter.infer_initial_value()
     
    def __repr__(self) -> str:
        return str(self.parameters)


class DescreteOptimizer:
    def __init__(self) -> None:
        pass


if __name__ == "__main__":
    p0_values = ListValues([0,1,2,3,4,5,6]) 
    p0 = Parameter("p0", {(None, None): p0_values})
    p0.infer_initial_value()
    print(f"current value of p0: {p0.current_value}")

    p1_values0 = ListValues([0,1,2,3]) 
    p1_values1 = ListValues([2,3,4,5]) 

    p1 = Parameter("p1", {(p0, ListValues([0,1,2])): p1_values0, (p0, ListValues([3,4,5])): p1_values1 })
    p1.infer_initial_value()
    print(f"current value of p1: {p1.current_value}")

    ps = ParameterSet([p0, p1])
    print(ps)

