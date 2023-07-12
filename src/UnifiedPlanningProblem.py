from timeit import default_timer as timer

import numpy as np
import unified_planning as up
from unified_planning.shortcuts import *


def compute_size_array(array: np.ndarray):
    size = 1
    for i in array.shape:
        size *= i
    return size


class UnifiedPlanningProblem:
    def __init__(
        self,
        horizon: int,
        ptdf: list[list],
        grid_params: dict,
        reference_states: dict,
    ):
        self.horizon = horizon
        self.ptdf = ptdf
        self.nb_buses = ptdf.shape[1]
        self.grid_params = grid_params
        self.nb_gens = len(grid_params["gens"]["pmax"])
        self.nb_storages = len(grid_params["storages"]["Emax"])
        self.reference_states = reference_states
        self.nb_transmission_lines = len(grid_params["lines"])
        self.slack_gens = np.where(grid_params["gens"]["slack"] == True)[0]

        start = timer()
        self._create_fluents()
        end = timer()
        print(f"Fluents creation: {end - start} seconds")

        start = timer()
        self._create_actions()
        end = timer()
        print(f"Actions creation: {end - start} seconds")

        start = timer()
        self._create_problem()
        end = timer()
        print(f"Problem creation: {end - start} seconds")

    def print_summary(self):
        print(
            f"Number of fluents: {compute_size_array(self.pgen)  + compute_size_array(self.congestions) + compute_size_array(self.flows)}"
        )
        print(f"Number of actions: {len(self.prod_target)}")

    def print_problem(self):
        print(self.problem)

    def _create_fluents(self):
        # Creating problem 'variables' so called fluents in PDDL
        self.pgen = np.array(
            [
                [Fluent(f"pgen_{gen_id}_{t}", RealType()) for t in range(self.horizon)]
                for gen_id in range(self.nb_gens)
            ]
        )

        # self.soc = np.array(
        #     [
        #         [
        #             Fluent(f"soc_{storage_id}_{t}", RealType(0, 1))
        #             for t in range(self.horizon)
        #         ]
        #         for storage_id in range(self.nb_storages)
        #     ]
        # )

        self.congestions = np.array(
            [
                [Fluent(f"congestion_{k}_{t}", BoolType()) for t in range(self.horizon)]
                for k in range(self.nb_transmission_lines)
            ]
        )

        self.flows = np.array(
            [
                [Fluent(f"flow_{k}_{t}", RealType()) for t in range(self.horizon)]
                for k in range(self.nb_transmission_lines)
            ]
        )

    def _create_actions(self):
        self.actions_costs = {}
        # Creating actions
        self.prod_target = []
        for gen_id in range(self.nb_gens):
            if (
                self.grid_params["gens"]["redispatchable"][gen_id] == True
                and self.grid_params["gens"]["slack"][gen_id] == False
            ):
                pmax = self.grid_params["gens"]["pmax"][gen_id]
                pmin = self.grid_params["gens"]["pmin"][gen_id]
                delta = int(pmax - pmin)

                for i in range(delta):
                    self.prod_target.append(
                        InstantaneousAction(f"prod_target_{gen_id}_{0}_{i}")
                    )
                    action = self.prod_target[-1]
                    self.actions_costs[action] = (
                        i * self.grid_params["gens"]["gen_cost_per_MW"][gen_id]
                    )
                    action.add_precondition(
                        GE(
                            float(self.reference_states["gens"][0][gen_id]),
                            i - self.grid_params["gens"]["max_ramp_up"][gen_id],
                        )
                    )
                    action.add_precondition(
                        LE(
                            float(self.reference_states["gens"][0][gen_id]),
                            i + self.grid_params["gens"]["max_ramp_down"][gen_id],
                        )
                    )
                    action.add_effect(self.pgen[gen_id][0], i)
                for k in range(self.nb_transmission_lines):
                    action.add_increase_effect(
                        self.flows[k][0],
                        float(self.ptdf[k][self.grid_params["gens"]["bus"][gen_id]])
                        * (
                            self.pgen[gen_id][0]
                            - float(self.reference_states["gens"][0][gen_id])
                        ),
                    )
                    action.add_effect(
                        self.congestions[k][0],
                        True,
                        condition=GE(self.flows[k][0], 5000),
                    )
                if len(self.slack_gens) > 1:
                    raise ("More than one slack generator!")
                else:
                    action.add_decrease_effect(
                        self.pgen[self.slack_gens[0]][0],
                        self.pgen[gen_id][0]
                        - float(self.reference_states["gens"][0][gen_id]),
                    )

                for t in range(1, self.horizon):
                    for i in range(delta):
                        self.prod_target.append(
                            InstantaneousAction(f"prod_target_{gen_id}_{t}_{i}")
                        )
                        action = self.prod_target[-1]
                        self.actions_costs[action] = (
                            i * self.grid_params["gens"]["gen_cost_per_MW"][gen_id]
                        )
                        action.add_precondition(
                            GE(
                                self.pgen[gen_id][t - 1],
                                i - self.grid_params["gens"]["max_ramp_up"][gen_id],
                            )
                        )
                        action.add_precondition(
                            LE(
                                self.pgen[gen_id][t - 1],
                                i + self.grid_params["gens"]["max_ramp_down"][gen_id],
                            )
                        )
                        action.add_effect(self.pgen[gen_id][t], i)
                    for k in range(self.nb_transmission_lines):
                        action.add_increase_effect(
                            self.flows[k][t],
                            float(self.ptdf[k][self.grid_params["gens"]["bus"][gen_id]])
                            * (
                                self.pgen[gen_id][t]
                                - float(self.reference_states["gens"][t][gen_id])
                            ),
                        )
                        action.add_effect(
                            self.congestions[k][t],
                            True,
                            condition=GE(
                                self.flows[k][t], 5000
                            ),  # TODO: add max flow on each line
                        )
                    if len(self.slack_gens) > 1:
                        raise ("More than one slack generator!")
                    else:
                        action.add_decrease_effect(
                            self.pgen[self.slack_gens[0]][t],
                            self.pgen[gen_id][t]
                            - float(self.reference_states["gens"][t][gen_id]),
                        )

        # self.soc_target = []
        # for storage_id in range(self.nb_storages):
        #     emax = self.grid_params["storages"]["Emax"][storage_id]
        #     emin = self.grid_params["storages"]["Emin"][storage_id]
        #     delta = int(emax - emin)
        #     for t in range(self.horizon - 1):
        #         for i in np.linspace(0, delta, delta * 10):
        #             self.soc_target.append(
        #                 InstantaneousAction(f"soc_target_{storage_id}_{t}_{i}")
        #             )
        #             action = self.soc_target[-1]
        #             self.actions_costs[action] = 0
        #             action.add_precondition(
        #                 GE(
        #                     self.soc[storage_id][t],
        #                     round(i, 2)
        #                     - self.grid_params["storages"]["Emax"][storage_id],
        #                 )
        #             )
        #             action.add_precondition(
        #                 LE(
        #                     self.soc[storage_id][t],
        #                     round(i, 2)
        #                     + self.grid_params["storages"]["Emin"][storage_id],
        #                 )
        #             )
        #             action.add_effect(self.soc[storage_id][t + 1], round(i, 2))
        #             # CONTINUE

    def _create_problem(self):
        problem = Problem("GridStability")

        # add fluents
        for gen_id in range(self.nb_gens):
            problem.add_fluents(self.pgen[gen_id])

        # for storage_id in range(self.nb_storages):
        #     problem.add_fluents(self.soc[storage_id])

        for k in range(self.nb_transmission_lines):
            problem.add_fluents(self.congestions[k])
            problem.add_fluents(self.flows[k])

        print("Fluents added")

        # add actions
        problem.add_actions(self.prod_target)
        # problem.add_actions(self.soc_target)

        print("Actions added")

        # add initial states
        for gen_id in range(self.nb_gens):
            for t in range(self.horizon):
                problem.set_initial_value(
                    self.pgen[gen_id][t],
                    float(self.reference_states["gens"][0][0]),
                )

        # for storage_id in range(self.nb_storages):
        #     problem.set_initial_value(
        #         self.soc[storage_id][0], float(self.init_states["storages"][storage_id])
        #     )

        for k in range(self.nb_transmission_lines):
            for t in range(self.horizon):
                problem.set_initial_value(self.congestions[k][t], False)
                i = self.grid_params["lines"][k]["from"]
                j = self.grid_params["lines"][k]["to"]
                problem.set_initial_value(
                    self.flows[k][t], float(self.reference_states["flows"][t][i][j])
                )

        print("Initial states added")

        problem.add_quality_metric(
            up.model.metrics.MinimizeActionCosts(self.actions_costs)
        )

        goal = And(
            [
                self.congestions[k][t]
                for k in range(self.nb_transmission_lines)
                for t in range(self.horizon)
            ]
        )
        problem.add_goal(goal)

        print("Objective added")

        self.problem = problem

    def solve(self):
        with OneshotPlanner(name="enhsp") as planner:
            print("Solving...")
            plan = planner.solve(self.problem).plan
            print("Plan found")
            print(plan)
            # with PlanValidator(
            #     problem_kind=self.problem.kind, plan_kind=plan.kind
            # ) as validator:
            #     validation = validator.validate(self.problem, plan)
            #     Valid = up.engines.ValidationResultStatus.VALID
            #     assert validation.status == Valid
            #     print("Plan is valid")
