import os
from os.path import join as pjoin

import numpy as np
from unified_planning.shortcuts import *

import config as cfg
from utils import compute_size_array


class UnifiedPlanningProblem:
    def __init__(
        self,
        horizon: int,
        ptdf: list[list],
        grid_params: dict,
        initial_states: dict,
        forecasted_states: dict,
        solver: str,
    ):
        self.horizon = horizon
        self.ptdf = ptdf
        self.grid_params = grid_params
        self.nb_gens = 2  # len(grid_params[cfg.GENERATORS][cfg.PMAX])
        self.nb_storages = len(grid_params[cfg.STORAGES][cfg.EMAX])
        self.initial_states = initial_states
        self.forecasted_states = forecasted_states
        self.nb_transmission_lines = 1  # len(grid_params[cfg.TRANSMISSION_LINES])
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
                and gen_id != 1
            ):
                pmax = self.grid_params[cfg.GENERATORS][cfg.PMAX][gen_id]
                pmin = self.grid_params[cfg.GENERATORS][cfg.PMIN][gen_id]
                delta = int(pmax - pmin)

                for i in range(80, 100):
                    self.prod_target.append(
                        InstantaneousAction(f"prod_target_{gen_id}_{0}_{i}")
                    )
                    action = self.prod_target[-1]
                    self.actions_costs[action] = (
                        abs(i - self.initial_states[cfg.GENERATORS][gen_id])
                        * self.grid_params[cfg.GENERATORS][cfg.GEN_COST_PER_MW][gen_id]
                    )
                    action.add_precondition(
                        GE(
                            float(self.initial_states[cfg.GENERATORS][gen_id]),
                            i
                            - self.grid_params[cfg.GENERATORS][cfg.MAX_RAMP_UP][gen_id],
                        )
                    )
                    action.add_precondition(
                        LE(
                            float(self.initial_states[cfg.GENERATORS][gen_id]),
                            i
                            + self.grid_params[cfg.GENERATORS][cfg.MAX_RAMP_DOWN][
                                gen_id
                            ],
                        )
                    )
                    action.add_precondition(
                        And(
                            GE(
                                self.pgen[gen_id][0],
                                float(self.forecasted_states[cfg.GENERATORS][gen_id][0])
                                - 10e-3,
                            ),
                            LE(
                                self.pgen[gen_id][0],
                                float(self.forecasted_states[cfg.GENERATORS][gen_id][0])
                                + 10e-3,
                            ),
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
                                    self.forecasted_states[cfg.GENERATORS][0][gen_id]
                                )
                            ),
                        )
                        action.add_effect(
                            self.congestions[k][0],
                            True,
                            condition=And(
                                GE(
                                    self.flows[k][0],
                                    float(
                                        self.grid_params[cfg.TRANSMISSION_LINES][k][
                                            cfg.MAX_FLOW
                                        ]
                                    ),
                                ),
                                LE(
                                    self.flows[k][0],
                                    float(
                                        -self.grid_params[cfg.TRANSMISSION_LINES][k][
                                            cfg.MAX_FLOW
                                        ]
                                    ),
                                ),
                            ),
                        )
                        action.add_effect(
                            self.congestions[k][0],
                            False,
                            condition=And(
                                LT(
                                    self.flows[k][0],
                                    float(
                                        self.grid_params[cfg.TRANSMISSION_LINES][k][
                                            cfg.MAX_FLOW
                                        ]
                                    ),
                                ),
                                GT(
                                    self.flows[k][0],
                                    float(
                                        -self.grid_params[cfg.TRANSMISSION_LINES][k][
                                            cfg.MAX_FLOW
                                        ]
                                    ),
                                ),
                            ),
                        )
                    if len(self.slack_gens) > 1:
                        raise ("More than one slack generator!")
                    else:
                        action.add_decrease_effect(
                            self.pgen[1][0],
                            self.pgen[gen_id][0]
                            - float(self.forecasted_states[cfg.GENERATORS][0][gen_id]),
                        )

                for t in range(1, self.horizon):
                    for i in range(delta):
                        self.prod_target.append(
                            InstantaneousAction(f"prod_target_{gen_id}_{t}_{i}")
                        )
                        action = self.prod_target[-1]
                        self.actions_costs[action] = (
                            abs(i - self.pgen[gen_id][t - 1])
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
                                        self.forecasted_states[cfg.GENERATORS][t][
                                            gen_id
                                        ]
                                    )
                                ),
                            )
                            action.add_effect(
                                self.congestions[k][t],
                                True,
                                condition=And(
                                    GE(
                                        self.flows[k][t],
                                        float(
                                            self.grid_params[cfg.TRANSMISSION_LINES][k][
                                                cfg.MAX_FLOW
                                            ]
                                        ),
                                    ),
                                    LE(
                                        self.flows[k][t],
                                        float(
                                            -self.grid_params[cfg.TRANSMISSION_LINES][
                                                k
                                            ][cfg.MAX_FLOW]
                                        ),
                                    ),
                                ),
                            )
                            action.add_effect(
                                self.congestions[k][t],
                                False,
                                condition=And(
                                    LT(
                                        self.flows[k][t],
                                        float(
                                            self.grid_params[cfg.TRANSMISSION_LINES][k][
                                                cfg.MAX_FLOW
                                            ]
                                        ),
                                    ),
                                    GT(
                                        self.flows[k][t],
                                        float(
                                            -self.grid_params[cfg.TRANSMISSION_LINES][
                                                k
                                            ][cfg.MAX_FLOW]
                                        ),
                                    ),
                                ),
                            )
                        if len(self.slack_gens) > 1:
                            raise ("More than one slack generator!")
                        else:
                            action.add_decrease_effect(
                                self.pgen[self.slack_gens[0]][t],
                                self.pgen[gen_id][t]
                                - float(
                                    self.forecasted_states[cfg.GENERATORS][t][gen_id]
                                ),
                            )

        # self.soc_target = []
        # for storage_id in range(self.nb_storages):
        #     emax = self.grid_params[cfg.STORAGES][cfg.EMAX][storage_id]
        #     emin = self.grid_params[cfg.STORAGES]["Emin"][storage_id]
        #     delta = int(emax - emin)

    def create_problem(self):
        problem = Problem("GridStability")

        # add fluents
        for gen_id in range(self.nb_gens):
            for t in range(self.horizon):
                problem.add_fluent(self.pgen[gen_id][t])

        # for storage_id in range(self.nb_storages):
        #     problem.add_fluents(self.soc[storage_id])

        for k in range(self.nb_transmission_lines):
            for t in range(self.horizon):
                problem.add_fluent(self.congestions[k][t])
                problem.add_fluent(self.flows[k][t])

        # add actions
        problem.add_actions(self.prod_target)
        # problem.add_actions(self.soc_target)

        # add initial states
        for gen_id in range(self.nb_gens):
            for t in range(self.horizon):
                problem.set_initial_value(
                    self.pgen[gen_id][t],
                    float(self.forecasted_states[cfg.GENERATORS][t][gen_id]),
                )

        # for storage_id in range(self.nb_storages):
        #     problem.set_initial_value(
        #         self.soc[storage_id][0], float(self.init_states[cfg.STORAGES][storage_id])
        #     )

        for k in range(self.nb_transmission_lines):
            for t in range(self.horizon):
                problem.set_initial_value(
                    self.congestions[k][t],
                    bool(self.forecasted_states[cfg.TRANSMISSION_LINES][t][k]),
                )
                problem.set_initial_value(
                    self.flows[k][t], float(self.forecasted_states[cfg.FLOWS][t][k])
                )

        problem.set_initial_value(self.flows[0][0], 125.0)
        problem.set_initial_value(self.congestions[0][0], True)

        # add quality metrics for optimization + goal
        problem.add_quality_metric(
            up.model.metrics.MinimizeActionCosts(self.actions_costs)
        )

        goals = [
            Not(self.congestions[k][t])
            for k in range(self.nb_transmission_lines)
            for t in range(self.horizon)
        ]

        problem.add_goal(And(goals))

        self.problem = problem

    def save_problem(self):
        os.makedirs(cfg.TMP_DIR, exist_ok=True)
        with open(pjoin(cfg.TMP_DIR, cfg.UP_PROBLEM), "w") as f:
            f.write(
                f"number of fluents: {compute_size_array(self.pgen)  + compute_size_array(self.congestions) + compute_size_array(self.flows)}\n"
            )
            f.write(f"number of actions: {len(self.prod_target)}\n")
            f.write(self.problem.__str__())
        f.close()

    def solve(self):
        with OneshotPlanner(name=self.solver) as planner:
            plan = planner.solve(self.problem)
            print(plan)
