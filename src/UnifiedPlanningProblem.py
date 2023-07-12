import os
from timeit import default_timer as timer

import numpy as np
import unified_planning as up
from unified_planning.shortcuts import *

import src.config as cfg
from src.utils import compute_size_array


class UnifiedPlanningProblem:
    def __init__(
        self,
        horizon: int,
        ptdf: list[list],
        grid_params: dict,
        reference_states: dict,
        solver: str,
    ):
        self.horizon = horizon
        self.ptdf = ptdf
        self.nb_buses = ptdf.shape[1]
        self.grid_params = grid_params
        self.nb_gens = len(grid_params[cfg.GENERATORS][cfg.PMAX])
        self.nb_storages = len(grid_params[cfg.STORAGES][cfg.EMAX])
        self.reference_states = reference_states
        self.nb_transmission_lines = len(grid_params[cfg.TRANSMISSION_LINES])
        self.slack_gens = np.where(grid_params[cfg.GENERATORS][cfg.SLACK] == True)[0]
        self.solver = solver

        self.create_fluents()
        self.create_actions()
        self.create_problem()

    def create_fluents(self):
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

    def create_actions(self):
        self.actions_costs = {}
        # Creating actions
        self.prod_target = []
        for gen_id in range(self.nb_gens):
            if (
                self.grid_params[cfg.GENERATORS][cfg.REDISPATCHABLE][gen_id] == True
                and self.grid_params[cfg.GENERATORS][cfg.SLACK][gen_id] == False
            ):
                pmax = self.grid_params[cfg.GENERATORS][cfg.PMAX][gen_id]
                pmin = self.grid_params[cfg.GENERATORS][cfg.PMIN][gen_id]
                delta = int(pmax - pmin)

                for i in range(delta):
                    self.prod_target.append(
                        InstantaneousAction(f"prod_target_{gen_id}_{0}_{i}")
                    )
                    action = self.prod_target[-1]
                    self.actions_costs[action] = (
                        i
                        * self.grid_params[cfg.GENERATORS][cfg.GEN_COST_PER_MW][gen_id]
                    )
                    action.add_precondition(
                        GE(
                            float(self.reference_states[cfg.GENERATORS][0][gen_id]),
                            i
                            - self.grid_params[cfg.GENERATORS][cfg.MAX_RAMP_UP][gen_id],
                        )
                    )
                    action.add_precondition(
                        LE(
                            float(self.reference_states[cfg.GENERATORS][0][gen_id]),
                            i
                            + self.grid_params[cfg.GENERATORS][cfg.MAX_RAMP_DOWN][
                                gen_id
                            ],
                        )
                    )
                    action.add_effect(self.pgen[gen_id][0], i)
                    for k in range(self.nb_transmission_lines):
                        action.add_increase_effect(
                            self.flows[k][0],
                            float(
                                self.ptdf[k][
                                    self.grid_params[cfg.GENERATORS][cfg.BUS][gen_id]
                                ]
                            )
                            * (
                                self.pgen[gen_id][0]
                                - float(
                                    self.reference_states[cfg.GENERATORS][0][gen_id]
                                )
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
                            - float(self.reference_states[cfg.GENERATORS][0][gen_id]),
                        )

                for t in range(1, self.horizon):
                    for i in range(delta):
                        self.prod_target.append(
                            InstantaneousAction(f"prod_target_{gen_id}_{t}_{i}")
                        )
                        action = self.prod_target[-1]
                        self.actions_costs[action] = (
                            i
                            * self.grid_params[cfg.GENERATORS][cfg.GEN_COST_PER_MW][
                                gen_id
                            ]
                        )
                        action.add_precondition(
                            GE(
                                self.pgen[gen_id][t - 1],
                                i
                                - self.grid_params[cfg.GENERATORS][cfg.MAX_RAMP_UP][
                                    gen_id
                                ],
                            )
                        )
                        action.add_precondition(
                            LE(
                                self.pgen[gen_id][t - 1],
                                i
                                + self.grid_params[cfg.GENERATORS][cfg.MAX_RAMP_DOWN][
                                    gen_id
                                ],
                            )
                        )
                        action.add_effect(self.pgen[gen_id][t], i)
                        for k in range(self.nb_transmission_lines):
                            action.add_increase_effect(
                                self.flows[k][t],
                                float(
                                    self.ptdf[k][
                                        self.grid_params[cfg.GENERATORS][cfg.BUS][
                                            gen_id
                                        ]
                                    ]
                                )
                                * (
                                    self.pgen[gen_id][t]
                                    - float(
                                        self.reference_states[cfg.GENERATORS][t][gen_id]
                                    )
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
                                - float(
                                    self.reference_states[cfg.GENERATORS][t][gen_id]
                                ),
                            )

        # self.soc_target = []
        # for storage_id in range(self.nb_storages):
        #     emax = self.grid_params[cfg.STORAGES][cfg.EMAX][storage_id]
        #     emin = self.grid_params[cfg.STORAGES]["Emin"][storage_id]
        #     delta = int(emax - emin)
        # TODO: add storage actions

    def create_problem(self):
        problem = Problem("GridStability")

        # add fluents
        for gen_id in range(self.nb_gens):
            problem.add_fluents(self.pgen[gen_id])

        # for storage_id in range(self.nb_storages):
        #     problem.add_fluents(self.soc[storage_id])

        for k in range(self.nb_transmission_lines):
            problem.add_fluents(self.congestions[k])
            problem.add_fluents(self.flows[k])

        # add actions
        problem.add_actions(self.prod_target)
        # problem.add_actions(self.soc_target)

        # add initial states
        for gen_id in range(self.nb_gens):
            for t in range(self.horizon):
                problem.set_initial_value(
                    self.pgen[gen_id][t],
                    float(self.reference_states[cfg.GENERATORS][0][0]),
                )

        # for storage_id in range(self.nb_storages):
        #     problem.set_initial_value(
        #         self.soc[storage_id][0], float(self.init_states[cfg.STORAGES][storage_id])
        #     )

        for k in range(self.nb_transmission_lines):
            for t in range(self.horizon):
                problem.set_initial_value(self.congestions[k][t], False)
                i = self.grid_params[cfg.TRANSMISSION_LINES][k][cfg.FROM_BUS]
                j = self.grid_params[cfg.TRANSMISSION_LINES][k][cfg.TO_BUS]
                problem.set_initial_value(
                    self.flows[k][t], float(self.reference_states[cfg.FLOWS][t][i][j])
                )

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

        self.problem = problem

    def save_problem(self):
        os.makedirs("tmp", exist_ok=True)
        with open("tmp/problem.up", "w") as f:
            f.write(
                f"number of fluents: {compute_size_array(self.pgen)  + compute_size_array(self.congestions) + compute_size_array(self.flows)}\n"
            )
            f.write(f"number of actions: {len(self.prod_target)}\n")
            f.write(self.problem.__str__())

    def solve(self):
        with OneshotPlanner(name=self.solver) as planner:
            plan = planner.solve(self.problem).plan
            # with PlanValidator(
            #     problem_kind=self.problem.kind, plan_kind=plan.kind
            # ) as validator:
            #     validation = validator.validate(self.problem, plan)
            #     Valid = up.engines.ValidationResultStatus.VALID
            #     assert validation.status == Valid
            #     print("Plan is valid")
