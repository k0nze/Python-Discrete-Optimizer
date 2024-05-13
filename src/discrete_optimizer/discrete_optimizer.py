import sys
import logging
import time
import random
import math
import numpy as np

from typing import Any, Callable, Dict, List, Optional, Tuple

from numpy.typing import NDArray


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


class ListValues(Values):
    def __init__(self, values: List[Any]) -> None:
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
        return key in self.values

    def get_first_value(self):
        return self.values[0]


class Parameter:
    def __init__(
        self,
        name: str,
        dependencies: Dict[Tuple[Optional["Parameter"], Optional[Values]], Values],
    ) -> None:
        self.name = name

        if type(dependencies) is list:
            self.dependencies = {(None, None): ListValues(dependencies)}
        elif isinstance(dependencies, Values):
            self.dependencies = {(None, None): dependencies}
        else:
            self.dependencies = dependencies

        self.current_value = None
        self.contrained_values = None

    def __iter__(self):
        self.constrain_values_via_dependencies()
        self.current_value = self.contrained_values[0]
        self.index = 0
        return self

    def __next__(self):
        if self.index < len(self.contrained_values):
            self.current_value = self.contrained_values[self.index]
            self.index += 1
            return self.current_value
        else:
            raise StopIteration

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

            self.constrain_values_via_dependencies()

            if len(self.contrained_values) == 0:
                self.current_value = None
            else:
                self.current_value = self.contrained_values[0]

    def constrain_values_via_dependencies(self):
        if (None, None) in self.dependencies.keys():
            self.contrained_values = list(self.dependencies[(None, None)])
        else:
            possible_values = set()

            for (
                constraining_parameter,
                contraining_value_range,
            ), possible_value_range in self.dependencies.items():
                # check if the current value of constraining parameter is in the constraining range
                # if this is the case add the values of the possible value range to values
                if constraining_parameter.current_value in contraining_value_range:
                    for value in possible_value_range:
                        possible_values.add(value)

            self.contrained_values = sorted(list(possible_values))


class ParameterSet:
    def __init__(self, *parameters, exclude: Optional[List[List[Any]]] = None) -> None:
        # CAUTION: parameters need to be ordered by their dependencies
        self.parameters = list(parameters)
        self.exclude = exclude
        self.initialize_parameters()

    def initialize_parameters(self) -> None:
        for parameter in self.parameters:
            parameter.infer_initial_value()

    def get_design_space(self) -> List[Tuple[Any, ...]]:
        design_space = [()]

        for index, current_parameter in enumerate(self.parameters):
            new_design_space = []
            for design_point in design_space:
                # set the curren_value of parameters on which the current_parameter might
                # depend to the to the value in the design point in order to constrain the
                # current_parameter correctly
                for i in range(index):
                    self.parameters[i].current_value = design_point[i]

                # when iterating over the values of a parameter it is automatically
                # contrained
                for value in current_parameter:
                    new_design_point = design_point + (value,)
                    new_design_space.append(new_design_point)
            design_space = new_design_space

        if self.exclude is not None:
            design_space = [
                design_point
                for design_point in design_space
                if design_point not in self.exclude
            ]

        return list(set(design_space))

    def __repr__(self) -> str:
        return str(self.parameters)


class DiscreteOptimizer:
    def __init__(
        self, parameter_set: ParameterSet, objective_function: Callable
    ) -> None:
        self.parameter_set = parameter_set
        self.objective_function = objective_function
        logging.basicConfig(
            level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
        )

    def minimize(
        self, verbose=False
    ) -> Tuple[Tuple[Any, ...], Dict[Tuple[Any, ...], Tuple[int, float]], int]:
        return (None,), dict(), 0

    def log_info(self, verbose, message):
        if verbose:
            logging.info(message)

    def log_error(self, message):
        logging.error(message)


class GlobalSearch(DiscreteOptimizer):
    def __init__(
        self,
        parameter_set: ParameterSet,
        objective_function: Callable[Tuple[Any, ...], Any],
    ) -> None:
        super().__init__(parameter_set, objective_function)

    def minimize(
        self, verbose=False
    ) -> Tuple[Tuple[Any, ...], Dict[Tuple[Any, ...], Tuple[int, float]], int]:
        design_space = self.parameter_set.get_design_space()
        results = {key: None for key in design_space}

        min_design_point = design_space[0]
        min_result = sys.maxsize
        steps = 0

        self.log_info(verbose, f"Starting GlobalSearch.minimize()")

        for design_point in design_space:
            self.log_info(
                verbose, f"Evaluating design point: {design_point}, step={steps}"
            )

            try:
                start_time = time.process_time()
                result = self.objective_function(design_point)
                end_time = time.process_time()
                runtime = end_time - start_time

                self.log_info(
                    verbose, f"Evaluation done: {design_point} -> {result}, t={runtime}"
                )
            except Exception as e:
                self.log_error(f"Error evaluating design point: {design_point}")
                result = e
                runtime = -1

            results[design_point] = (result, runtime)

            if result < min_result:
                min_design_point = design_point
                min_result = result
                self.log_info(
                    verbose, f"Found new minimum: {min_design_point} -> {min_result}"
                )

            steps += 1

        self.log_info(
            verbose,
            f"Finished GlobalSearch.minimize(): {min_design_point} -> {min_result}",
        )

        return min_design_point, results, steps


class SimulatedAnnealing(DiscreteOptimizer):
    def __init__(
        self, parameter_set: ParameterSet, objective_function: Callable
    ) -> None:
        super().__init__(parameter_set, objective_function)

    @staticmethod
    def convert_design_space_to_numpy(design_space):
        return [np.array(x) for x in design_space]

    @staticmethod
    def euclidean_distance(p: Tuple[int, ...], q: Tuple[int, ...]) -> float:
        p_np = np.array(p)
        q_np = np.array(q)
        return np.linalg.norm(p_np - q_np)

    @staticmethod
    def pertubate(
        design_point: NDArray, design_space: List[NDArray], max_distance: float
    ) -> NDArray:
        while True:
            candidate_design_point = random.choice(design_space)
            if (
                not np.array_equal(candidate_design_point, design_point)
                and SimulatedAnnealing.euclidean_distance(
                    design_point, candidate_design_point
                )
                < max_distance
            ):
                return candidate_design_point

    @staticmethod
    def accept(delta_E, T):
        if delta_E < 0:
            return True
        else:
            # generate random number between [0,1)
            r = random.random()
            if r < math.exp(-delta_E / T):
                return True
            else:
                return False

    def minimize(
        self,
        initial_design_point=None,
        T_max=100,
        T_min=0.001,
        E_th=0,
        alpha=0.85,
        pertubation_function: Optional[Callable] = None,
        max_distance: float = 1.5,
        verbose=False,
    ) -> Tuple[Tuple[Any, ...], Dict[Tuple[Any, ...], Tuple[int, float]], int]:
        design_space = self.parameter_set.get_design_space()
        results = {key: None for key in design_space}

        if pertubation_function is None:
            pertubation_function = SimulatedAnnealing.pertubate

        np_design_space = SimulatedAnnealing.convert_design_space_to_numpy(design_space)

        if initial_design_point is None:
            design_point = np_design_space[0]

        min_design_point = design_point
        min_result = sys.maxsize
        steps = 0

        self.log_info(verbose, f"Starting GlobalSearch.minimize()")

        T = T_max
        E = self.objective_function(design_point)

        while T > T_min and E > E_th:
            new_design_point = SimulatedAnnealing.pertubate(
                design_point, design_space, max_distance
            )

            # check if design point was already evaluated
            if results[tuple(new_design_point)] is not None:
                self.log_info(
                    verbose, f"Using cached design point: {design_point}, step={steps}"
                )
                E_new = results[tuple(new_design_point)][0]

                # skip point if it produced an error perviously
                if isinstance(E_new, Exception):
                    continue
            else:
                self.log_info(
                    verbose, f"Evaluating design point: {design_point}, step={steps}"
                )
                try:
                    start_time = time.process_time()
                    E_new = self.objective_function(new_design_point)
                    end_time = time.process_time()
                    runtime = end_time - start_time

                    self.log_info(
                        verbose,
                        f"Evaluation done: {new_design_point} -> {E_new}, t={runtime}",
                    )
                except Exception as e:
                    self.log_error(f"Error evaluating design point: {design_point}")
                    E_new = e
                    runtime = -1

                    results[tuple(new_design_point)] = (E_new, runtime)

            if E_new < min_result:
                min_design_point = new_design_point
                min_result = E_new
                self.log_info(
                    verbose, f"Found new minimum: {min_design_point} -> {min_result}"
                )

            delta_E = E_new - E

            if SimulatedAnnealing.accept(delta_E, T):
                design_point = new_design_point
                E = E_new
            T = T * alpha

            steps += 1

        self.log_info(
            verbose,
            f"Finished SimulatedAnnealing.minimize(): {min_design_point} -> {min_result}",
        )

        return tuple(min_design_point), results, steps
