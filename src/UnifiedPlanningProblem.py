import os
from os.path import join as pjoin

import numpy as np
import unified_planning as up
from unified_planning.engines import PlanGenerationResultStatus
from unified_planning.engines.sequential_simulator import (
    evaluate_quality_metric,
    evaluate_quality_metric_in_initial_state,
)
from unified_planning.shortcuts import *

import config as cfg
from utils import compute_size_array
from unified_planning.engines.sequential_simulator import (
    evaluate_quality_metric,
    evaluate_quality_metric_in_initial_state,
)


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
        get_environment().credits_stream = None
        self.horizon = horizon
        self.ptdf = ptdf
        self.grid_params = grid_params
        self.nb_gens = 2
        self.nb_storages = len(grid_params[cfg.STORAGES][cfg.EMAX])
        self.initial_states = initial_states
        self.forecasted_states = forecasted_states
        self.nb_transmission_lines = 1
        self.slack_gens = np.where(grid_params[cfg.GENERATORS][cfg.SLACK] == True)[0]
        self.solver = solver

        self.create_fluents()
        self.create_gen_actions()
        self.create_problem()

    def create_fluents(self):
        # Creating problem 'variables' so called fluents in PDDL
        self.pgen = np.array(
            [
                [Fluent(f"pgen_{gen_id}_{t}", RealType()) for t in range(self.horizon)]
                for gen_id in range(self.nb_gens)
            ]
        )

        self.pgen_exp = np.array(
            [
                [FluentExp(self.pgen[gen_id][t]) for t in range(self.horizon)]
                for gen_id in range(self.nb_gens)
            ]
        )

        self.congestions = np.array(
            [
                [Fluent(f"congestion_{k}_{t}", BoolType()) for t in range(self.horizon)]
                for k in range(self.nb_transmission_lines)
            ]
        )

        self.congestions_exp = np.array(
            [
                [FluentExp(self.congestions[k][t]) for t in range(self.horizon)]
                for k in range(self.nb_transmission_lines)
            ]
        )

        self.flows = np.array(
            [
                [Fluent(f"flow_{k}_{t}", RealType()) for t in range(self.horizon)]
                for k in range(self.nb_transmission_lines)
            ]
        )

        self.flows_exp = np.array(
            [
                [FluentExp(self.flows[k][t]) for t in range(self.horizon)]
                for k in range(self.nb_transmission_lines)
            ]
        )

    def create_gen_actions(self):
        self.actions_costs = {}

        # Creating actions
        self.prod_target = []

        # for horizon 0
        for gen_id in range(self.nb_gens):
            if (
                self.grid_params[cfg.GENERATORS][cfg.REDISPATCHABLE][gen_id] == True
                and self.grid_params[cfg.GENERATORS][cfg.SLACK][gen_id] == False
                and gen_id != 1
            ):
                pmax = self.grid_params[cfg.GENERATORS][cfg.PMAX][gen_id]
                pmin = self.grid_params[cfg.GENERATORS][cfg.PMIN][gen_id]
                delta = int(pmax - pmin)

                for i in range(82, 85):
                    self.prod_target.append(
                        InstantaneousAction(f"prod_target_{gen_id}_{0}_{i}")
                    )
                    action = self.prod_target[-1]
                    self.actions_costs[action] = float(
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
                                float(
                                    self.forecasted_states[cfg.GENERATORS][0][gen_id]
                                    - 10e-3
                                ),
                            ),
                            LE(
                                self.pgen[gen_id][0],
                                float(
                                    self.forecasted_states[cfg.GENERATORS][0][gen_id]
                                    + 10e-3
                                ),
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
                                i
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
                                    self.flows[k][0]
                                    + float(
                                        self.ptdf[k][
                                            self.grid_params[cfg.GENERATORS][cfg.BUS][
                                                gen_id
                                            ]
                                        ]
                                    )
                                    * (
                                        i
                                        - float(
                                            self.forecasted_states[cfg.GENERATORS][0][
                                                gen_id
                                            ]
                                        )
                                    ),
                                    float(
                                        self.grid_params[cfg.TRANSMISSION_LINES][k][
                                            cfg.MAX_FLOW
                                        ]
                                    ),
                                ),
                                LE(
                                    self.flows[k][0]
                                    + float(
                                        self.ptdf[k][
                                            self.grid_params[cfg.GENERATORS][cfg.BUS][
                                                gen_id
                                            ]
                                        ]
                                    )
                                    * (
                                        i
                                        - float(
                                            self.forecasted_states[cfg.GENERATORS][0][
                                                gen_id
                                            ]
                                        )
                                    ),
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
                                    self.flows[k][0]
                                    + float(
                                        self.ptdf[k][
                                            self.grid_params[cfg.GENERATORS][cfg.BUS][
                                                gen_id
                                            ]
                                        ]
                                    )
                                    * (
                                        i
                                        - float(
                                            self.forecasted_states[cfg.GENERATORS][0][
                                                gen_id
                                            ]
                                        )
                                    ),
                                    float(
                                        self.grid_params[cfg.TRANSMISSION_LINES][k][
                                            cfg.MAX_FLOW
                                        ]
                                    ),
                                ),
                                GT(
                                    self.flows[k][0]
                                    + float(
                                        self.ptdf[k][
                                            self.grid_params[cfg.GENERATORS][cfg.BUS][
                                                gen_id
                                            ]
                                        ]
                                    )
                                    * (
                                        i
                                        - float(
                                            self.forecasted_states[cfg.GENERATORS][0][
                                                gen_id
                                            ]
                                        )
                                    ),
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
                            i
                            - float(self.forecasted_states[cfg.GENERATORS][0][gen_id]),
                        )

                # for horizon > 1
                # TODO

    def create_problem(self):
        problem = Problem("GridStability")

        # add fluents
        for gen_id in range(self.nb_gens):
            for t in range(self.horizon):
                problem.add_fluent(self.pgen[gen_id][t])

        for k in range(self.nb_transmission_lines):
            for t in range(self.horizon):
                problem.add_fluent(self.congestions[k][t])
                problem.add_fluent(self.flows[k][t])

        # add actions
        problem.add_actions(self.prod_target)

        # add initial states
        for gen_id in range(self.nb_gens):
            for t in range(self.horizon):
                problem.set_initial_value(
                    self.pgen[gen_id][t],
                    float(self.forecasted_states[cfg.GENERATORS][t][gen_id]),
                )

        for k in range(self.nb_transmission_lines):
            for t in range(self.horizon):
                problem.set_initial_value(
                    self.congestions[k][t],
                    bool(self.forecasted_states[cfg.TRANSMISSION_LINES][t][k]),
                )
                problem.set_initial_value(
                    self.flows[k][t], float(self.forecasted_states[cfg.FLOWS][t][k])
                )

        problem.set_initial_value(self.flows[0][0], 120)
        problem.set_initial_value(self.congestions[0][0], True)

        # add quality metrics for optimization + goal
        self.quality_metric = up.model.metrics.MinimizeActionCosts(self.actions_costs)
        problem.add_quality_metric(self.quality_metric)

        goals = [
            Not(self.congestions[k][t])
            for k in range(self.nb_transmission_lines)
            for t in range(self.horizon)
        ]

        problem.add_goal(And(goals))

        self.problem = problem

    def save_problem(self):
        os.makedirs(cfg.TMP_DIR, exist_ok=True)

        # upp problem, "upp" stands for unified planning problem
        with open(pjoin(cfg.TMP_DIR, cfg.UPP_PROBLEM), "w") as f:
            f.write(
                f"number of fluents: {compute_size_array(self.pgen)  + compute_size_array(self.congestions) + compute_size_array(self.flows)}\n"
            )
            f.write(f"number of actions: {len(self.prod_target)}\n")
            f.write(self.problem.__str__())
        f.close()

        # pddl problem
        up.io.PDDLWriter(self.problem, True, True).write_problem(
            pjoin(cfg.TMP_DIR, cfg.PDDL_PROBLEM)
        )

    def solve(self):
        with OneshotPlanner(
            name=self.solver,
            optimality_guarantee=PlanGenerationResultStatus.SOLVED_OPTIMALLY,
        ) as planner:
            output = planner.solve(self.problem)
            plan = output.plan
            if plan is None:
                print(output)
            if plan is not None:
                print(f"Status: {output.status}")
                print(f"Plan found: {plan}")
                print("Simulating plan...")
                with SequentialSimulator(problem=self.problem) as simulator:
                    initial_state = simulator.get_initial_state()
                    minimize_cost_value = evaluate_quality_metric_in_initial_state(
                        simulator, self.quality_metric
                    )
                    states = [initial_state]
                    for ai in plan.actions:
                        print(f"\taction: {ai}")
                        state_test = simulator.apply(initial_state, ai)
                        states.append(state_test)
                        minimize_cost_value = evaluate_quality_metric(
                            simulator,
                            self.quality_metric,
                            minimize_cost_value,
                            initial_state,
                            ai.action,
                            ai.actual_parameters,
                            state_test,
                        )
                        print(f"\tcost: {float(minimize_cost_value)}")
