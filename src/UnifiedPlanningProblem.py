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
from utils import compute_size_array, setup_logger


class UnifiedPlanningProblem:
    def __init__(
        self,
        operational_horizon: int,
        ptdf: list[list],
        grid_params: dict,
        initial_states: dict,
        forecasted_states: dict,
        solver: str,
        problem_id: int,
    ):
        get_environment().credits_stream = None

        self.operational_horizon = operational_horizon
        self.ptdf = ptdf
        self.grid_params = grid_params
        self.nb_gens = len(grid_params[cfg.GENERATORS][cfg.PMAX])
        self.nb_storages = len(grid_params[cfg.STORAGES][cfg.EMAX])
        self.initial_states = initial_states
        self.forecasted_states = forecasted_states
        self.nb_transmission_lines = len(grid_params[cfg.TRANSMISSION_LINES])
        self.slack_gens = np.where(grid_params[cfg.GENERATORS][cfg.SLACK] == True)[0]
        self.solver = solver
        self.id = problem_id

        self.float_precision = 6

        self.log_dir = pjoin(cfg.LOG_DIR, f"problem_{self.id}")
        os.makedirs(self.log_dir, exist_ok=True)

        self.logger = setup_logger(
            f"{__name__}_{self.id}",
            self.log_dir,
        )

        self.create_fluents()
        self.create_actions()
        self.create_problem()

    def create_fluents(self):
        # Creating problem 'variables' so called fluents in PDDL
        self.pgen = np.array(
            [
                [
                    Fluent(f"pgen_{gen_id}_{t}", RealType())
                    for t in range(self.operational_horizon)
                ]
                for gen_id in range(self.nb_gens)
            ]
        )

        self.pgen_exp = np.array(
            [
                [
                    FluentExp(self.pgen[gen_id][t])
                    for t in range(self.operational_horizon)
                ]
                for gen_id in range(self.nb_gens)
            ]
        )

        self.congestions = np.array(
            [
                [
                    Fluent(f"congestion_{k}_{t}", BoolType())
                    for t in range(self.operational_horizon)
                ]
                for k in range(self.nb_transmission_lines)
            ]
        )

        self.congestions_exp = np.array(
            [
                [
                    FluentExp(self.congestions[k][t])
                    for t in range(self.operational_horizon)
                ]
                for k in range(self.nb_transmission_lines)
            ]
        )

        self.flows = np.array(
            [
                [
                    Fluent(f"flow_{k}_{t}", RealType())
                    for t in range(self.operational_horizon)
                ]
                for k in range(self.nb_transmission_lines)
            ]
        )

        self.flows_exp = np.array(
            [
                [FluentExp(self.flows[k][t]) for t in range(self.operational_horizon)]
                for k in range(self.nb_transmission_lines)
            ]
        )

        self.update_status = np.array(
            [
                Fluent(f"update_status_{t}", BoolType())
                for t in range(self.operational_horizon)
            ]
        )

    def create_gen_actions(self) -> dict:
        actions_costs = {}

        # Creating actions
        self.pgen_actions = []

        # for horizon 0
        for gen_id in range(self.nb_gens):
            if (
                self.grid_params[cfg.GENERATORS][cfg.REDISPATCHABLE][gen_id] == True
                and self.grid_params[cfg.GENERATORS][cfg.SLACK][gen_id] == False
            ):
                pmax = int(self.grid_params[cfg.GENERATORS][cfg.PMAX][gen_id])
                pmin = int(self.grid_params[cfg.GENERATORS][cfg.PMIN][gen_id])

                for i in range(pmin, pmax + 1):
                    if not (
                        float(self.initial_states[cfg.GENERATORS][gen_id])
                        >= i - self.grid_params[cfg.GENERATORS][cfg.MAX_RAMP_UP][gen_id]
                        and float(self.initial_states[cfg.GENERATORS][gen_id])
                        <= i
                        + self.grid_params[cfg.GENERATORS][cfg.MAX_RAMP_DOWN][gen_id]
                    ):
                        continue
                    self.pgen_actions.append(
                        InstantaneousAction(f"gen_target_{gen_id}_{0}_{i}")
                    )
                    action = self.pgen_actions[-1]
                    actions_costs[action] = float(
                        abs(i - self.initial_states[cfg.GENERATORS][gen_id])
                        * self.grid_params[cfg.GENERATORS][cfg.GEN_COST_PER_MW][gen_id]
                    )
                    action.add_precondition(Iff(self.update_status[0], False))
                    action.add_precondition(
                        And(
                            GE(
                                self.pgen[gen_id][0],
                                float(
                                    self.forecasted_states[cfg.GENERATORS][0][gen_id]
                                    - self.float_precision
                                ),
                            ),
                            LE(
                                self.pgen[gen_id][0],
                                float(
                                    self.forecasted_states[cfg.GENERATORS][0][gen_id]
                                    + self.float_precision
                                ),
                            ),
                        )
                    )
                    action.add_effect(self.pgen[gen_id][0], i)
                    for k in range(self.nb_transmission_lines):
                        diff = float(
                            round(
                                self.ptdf[k][
                                    self.grid_params[cfg.GENERATORS][cfg.BUS][gen_id]
                                ]
                                * (
                                    i
                                    - float(
                                        self.forecasted_states[cfg.GENERATORS][0][
                                            gen_id
                                        ]
                                    )
                                ),
                                self.float_precision,
                            )
                        )
                        action.add_increase_effect(
                            self.flows[k][0],
                            diff,
                        )
                        action.add_effect(
                            self.congestions[k][0],
                            True,
                            condition=Or(
                                GE(
                                    self.flows[k][0] + diff,
                                    float(
                                        self.grid_params[cfg.TRANSMISSION_LINES][k][
                                            cfg.MAX_FLOW
                                        ]
                                    ),
                                ),
                                LE(
                                    self.flows[k][0] + diff,
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
                                    self.flows[k][0] + diff,
                                    float(
                                        self.grid_params[cfg.TRANSMISSION_LINES][k][
                                            cfg.MAX_FLOW
                                        ]
                                    ),
                                ),
                                GT(
                                    self.flows[k][0] + diff,
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
                            self.pgen[self.slack_gens[-1]][0],
                            i
                            - float(self.forecasted_states[cfg.GENERATORS][0][gen_id]),
                        )
                # for horizon > 1
                # TODO
        return actions_costs

    def create_update_status_actions(self):
        actions_costs = {}
        self.update_status_actions = []
        for t in range(self.operational_horizon):
            self.update_status_actions.append(
                InstantaneousAction(f"update_status_{t}_true")
            )
            action = self.update_status_actions[-1]
            action.add_precondition(Iff(self.update_status[t], False))
            action.add_effect(self.update_status[t], True)
            actions_costs[action] = 0
        return actions_costs

    def create_actions(self):
        gen_costs = self.create_gen_actions()
        update_status_costs = self.create_update_status_actions()
        self.actions_costs = {**gen_costs, **update_status_costs}

    def create_problem(self):
        problem = Problem(f"GridStability_{self.id}")

        # add fluents
        for gen_id in range(self.nb_gens):
            for t in range(self.operational_horizon):
                problem.add_fluent(self.pgen[gen_id][t])

        for k in range(self.nb_transmission_lines):
            for t in range(self.operational_horizon):
                problem.add_fluent(self.congestions[k][t])
                problem.add_fluent(self.flows[k][t])

        for t in range(self.operational_horizon):
            problem.add_fluent(self.update_status[t])

        # add actions
        problem.add_actions(self.pgen_actions)
        problem.add_actions(self.update_status_actions)

        # add initial states
        for gen_id in range(self.nb_gens):
            for t in range(self.operational_horizon):
                problem.set_initial_value(
                    self.pgen[gen_id][t],
                    float(self.forecasted_states[cfg.GENERATORS][t][gen_id]),
                )

        for k in range(self.nb_transmission_lines):
            for t in range(self.operational_horizon):
                problem.set_initial_value(
                    self.congestions[k][t],
                    bool(self.forecasted_states[cfg.TRANSMISSION_LINES][t][k]),
                )
                problem.set_initial_value(
                    self.flows[k][t],
                    float(
                        round(
                            self.forecasted_states[cfg.FLOWS][t][k],
                            self.float_precision,
                        )
                    ),
                )

        for t in range(self.operational_horizon):
            problem.set_initial_value(self.update_status[t], False)

        # add quality metrics for optimization + goal
        self.quality_metric = up.model.metrics.MinimizeActionCosts(self.actions_costs)
        problem.add_quality_metric(self.quality_metric)

        goal_1 = [
            Iff(self.congestions[k][t], False)
            for k in range(self.nb_transmission_lines)
            for t in range(self.operational_horizon)
        ]  # is it too restrictive?

        goal_2 = [
            Iff(self.update_status[t], True) for t in range(self.operational_horizon)
        ]

        goals = goal_1 + goal_2

        problem.add_goal(
            And(goal_1)
        )  # TODO: finish the implementation of the update_status actions

        self.problem = problem

    def save_problem(self):
        upp_file = "problem_" + str(self.id) + cfg.UPP_SUFFIX
        pddl_file = "problem_" + str(self.id) + cfg.PDDL_SUFFIX
        pddl_domain_file = "problem_domain_" + str(self.id) + cfg.PDDL_SUFFIX

        # upp problem, "upp" stands for unified planning problem
        with open(pjoin(self.log_dir, upp_file), "w") as f:
            f.write(
                f"number of fluents: {compute_size_array(self.pgen) + compute_size_array(self.congestions) + compute_size_array(self.flows) + compute_size_array(self.update_status)}\n"
            )
            f.write(f"number of actions: {len(self.pgen_actions)}\n")
            f.write(self.problem.__str__())
        f.close()

        # pddl problem
        pddl_writer = up.io.PDDLWriter(self.problem, True, True)
        pddl_writer.write_problem(pjoin(self.log_dir, pddl_file))
        pddl_writer.write_domain(pjoin(self.log_dir, pddl_domain_file))

    def solve(self, simulate=False):
        with OneshotPlanner(
            name=self.solver,
            problem_kind=self.problem.kind,
            optimality_guarantee=PlanGenerationResultStatus.SOLVED_OPTIMALLY,
        ) as planner:
            output = planner.solve(self.problem)
            plan = output.plan
            if plan is None:
                self.logger.error(output)
                raise Exception("\tNo plan found!")
            else:
                self.logger.info(f"Status: {output.status}")
                self.logger.info(f"Plan found: {plan}\n")
                if simulate and len(plan.actions) > 0:
                    self.logger.debug("Simulating plan...")
                    with SequentialSimulator(problem=self.problem) as simulator:
                        initial_state = simulator.get_initial_state()
                        minimize_cost_value = evaluate_quality_metric_in_initial_state(
                            simulator, self.quality_metric
                        )
                        states = [initial_state]
                        for act in plan.actions:
                            self.logger.debug(f"\taction: {act}")
                            if act.action.name.startswith("update_status"):
                                self.logger.debug("\tupdate status new value: True")
                                continue
                            state_test = simulator.apply(initial_state, act)
                            states.append(state_test)
                            self.logger.debug(
                                f"\tgens new value: {[[float(state_test.get_value(self.pgen_exp[g][t]).constant_value()) for g in range(self.nb_gens)] for t in range(self.operational_horizon)]}"
                            )
                            self.logger.debug(
                                f"\tflows new value: {[[float(state_test.get_value(self.flows_exp[k][t]).constant_value()) for k in range(self.nb_transmission_lines)] for t in range(self.operational_horizon)]}"
                            )
                            self.logger.debug(
                                f"\tcongestions new value: {[[state_test.get_value(self.congestions_exp[k][t]) for k in range(self.nb_transmission_lines)] for t in range(self.operational_horizon)]}"
                            )
                            self.logger.debug(
                                f"\tgen slack new value: {[float(state_test.get_value(self.pgen_exp[self.slack_gens[0]][t]).constant_value()) for t in range(self.operational_horizon)]}"
                            )
                            minimize_cost_value = evaluate_quality_metric(
                                simulator,
                                self.quality_metric,
                                minimize_cost_value,
                                initial_state,
                                act.action,
                                act.actual_parameters,
                                state_test,
                            )
                            self.logger.debug(f"\tcost: {float(minimize_cost_value)}\n")
                return plan.actions
