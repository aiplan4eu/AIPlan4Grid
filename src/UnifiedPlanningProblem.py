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
from utils import compute_size_array, verbose_print


class UnifiedPlanningProblem:
    def __init__(
        self,
        tactical_horizon: int,
        ptdf: list[list],
        grid_params: dict,
        initial_states: dict,
        forecasted_states: dict,
        solver: str,
        verbose: bool,
    ):
        get_environment().credits_stream = None
        self.tactical_horizon = tactical_horizon
        self.ptdf = ptdf
        self.grid_params = grid_params
        self.nb_gens = len(grid_params[cfg.GENERATORS][cfg.PMAX])
        self.nb_storages = len(grid_params[cfg.STORAGES][cfg.EMAX])
        self.initial_states = initial_states
        self.forecasted_states = forecasted_states
        self.nb_transmission_lines = 18
        self.slack_gens = np.where(grid_params[cfg.GENERATORS][cfg.SLACK] == True)[0]
        self.solver = solver

        self.create_fluents()
        self.create_actions()
        self.create_problem()

        global vprint
        vprint = verbose_print(verbose)

    def create_fluents(self):
        # Creating problem 'variables' so called fluents in PDDL
        self.pgen = np.array(
            [
                [
                    Fluent(f"pgen_{gen_id}_{t}", RealType())
                    for t in range(self.tactical_horizon)
                ]
                for gen_id in range(self.nb_gens)
            ]
        )

        self.pgen_exp = np.array(
            [
                [FluentExp(self.pgen[gen_id][t]) for t in range(self.tactical_horizon)]
                for gen_id in range(self.nb_gens)
            ]
        )

        self.congestions = np.array(
            [
                [
                    Fluent(f"congestion_{k}_{t}", BoolType())
                    for t in range(self.tactical_horizon)
                ]
                for k in range(self.nb_transmission_lines)
            ]
        )

        self.congestions_exp = np.array(
            [
                [
                    FluentExp(self.congestions[k][t])
                    for t in range(self.tactical_horizon)
                ]
                for k in range(self.nb_transmission_lines)
            ]
        )

        self.flows = np.array(
            [
                [
                    Fluent(f"flow_{k}_{t}", RealType())
                    for t in range(self.tactical_horizon)
                ]
                for k in range(self.nb_transmission_lines)
            ]
        )

        self.flows_exp = np.array(
            [
                [FluentExp(self.flows[k][t]) for t in range(self.tactical_horizon)]
                for k in range(self.nb_transmission_lines)
            ]
        )

        self.update_status = np.array(
            [
                [
                    Fluent(f"update_status_{k}_{t}", BoolType())
                    for t in range(self.tactical_horizon)
                ]
                for k in range(self.nb_transmission_lines)
            ]
        )

        self.update_status_exp = np.array(
            [
                [
                    FluentExp(self.update_status[k][t])
                    for t in range(self.tactical_horizon)
                ]
                for k in range(self.nb_transmission_lines)
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
                pmax = self.grid_params[cfg.GENERATORS][cfg.PMAX][gen_id]
                pmin = self.grid_params[cfg.GENERATORS][cfg.PMIN][gen_id]
                delta = int(pmax - pmin)

                for i in range(delta):
                    self.pgen_actions.append(
                        InstantaneousAction(f"gen_target_{gen_id}_{0}_{i}")
                    )
                    action = self.pgen_actions[-1]
                    actions_costs[action] = float(
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
                        action.add_precondition(Iff(self.update_status[k][0], False))
                        diff = float(
                            self.ptdf[k][
                                self.grid_params[cfg.GENERATORS][cfg.BUS][gen_id]
                            ]
                        ) * (
                            i - float(self.forecasted_states[cfg.GENERATORS][0][gen_id])
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
                        action.add_effect(self.update_status[k][0], True)
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

    def create_actions(self):
        gen_costs = self.create_gen_actions()
        self.actions_costs = {**gen_costs}

    def create_problem(self):
        problem = Problem("GridStability")

        # add fluents
        for gen_id in range(self.nb_gens):
            for t in range(self.tactical_horizon):
                problem.add_fluent(self.pgen[gen_id][t])

        for k in range(self.nb_transmission_lines):
            for t in range(self.tactical_horizon):
                problem.add_fluent(self.congestions[k][t])
                problem.add_fluent(self.flows[k][t])
                problem.add_fluent(self.update_status[k][t])

        # add actions
        problem.add_actions(self.pgen_actions)

        # add initial states
        for gen_id in range(self.nb_gens):
            for t in range(self.tactical_horizon):
                problem.set_initial_value(
                    self.pgen[gen_id][t],
                    float(self.forecasted_states[cfg.GENERATORS][t][gen_id]),
                )

        for k in range(self.nb_transmission_lines):
            for t in range(self.tactical_horizon):
                problem.set_initial_value(
                    self.congestions[k][t],
                    bool(self.forecasted_states[cfg.TRANSMISSION_LINES][t][k]),
                )
                problem.set_initial_value(
                    self.flows[k][t], float(self.forecasted_states[cfg.FLOWS][t][k])
                )
                problem.set_initial_value(self.update_status[k][t], False)

        problem.set_initial_value(self.flows[6][0], 75)
        problem.set_initial_value(self.congestions[6][0], True)

        # add quality metrics for optimization + goal
        self.quality_metric = up.model.metrics.MinimizeActionCosts(self.actions_costs)
        problem.add_quality_metric(self.quality_metric)

        goals = [
            Iff(self.congestions[k][t], False)
            for k in range(self.nb_transmission_lines)
            for t in range(self.tactical_horizon)
        ]  # is it too restrictive?

        problem.add_goal(And(goals))

        self.problem = problem

    def save_problem(self, id: int):
        save_dir = pjoin(cfg.TMP_DIR, f"problem_{id}")
        os.makedirs(save_dir, exist_ok=True)
        upp_file = str(id) + "_problem" + cfg.UPP_SUFFIX
        pddl_file = str(id) + "_problem" + cfg.PDDL_SUFFIX
        pddl_domain_file = str(id) + "_problem_domain" + cfg.PDDL_SUFFIX

        # upp problem, "upp" stands for unified planning problem
        with open(pjoin(save_dir, upp_file), "w") as f:
            f.write(
                f"number of fluents: {compute_size_array(self.pgen)  + compute_size_array(self.congestions) + compute_size_array(self.flows)}\n"
            )
            f.write(f"number of actions: {len(self.pgen_actions)}\n")
            f.write(self.problem.__str__())
        f.close()

        # pddl problem
        pddl_writer = up.io.PDDLWriter(self.problem, True, True)
        pddl_writer.write_problem(pjoin(save_dir, pddl_file))
        pddl_writer.write_domain(pjoin(save_dir, pddl_domain_file))

    def solve(self, simulate=False):
        with OneshotPlanner(
            name=self.solver,
            problem_kind=self.problem.kind,
            optimality_guarantee=PlanGenerationResultStatus.SOLVED_OPTIMALLY,
        ) as planner:
            output = planner.solve(self.problem)
            plan = output.plan
            if plan is None:
                vprint(output)
            else:
                vprint(f"Status: {output.status}")
                vprint(f"Plan found: {plan}")
                vprint("Simulating plan...")
                if simulate:
                    with SequentialSimulator(problem=self.problem) as simulator:
                        initial_state = simulator.get_initial_state()
                        minimize_cost_value = evaluate_quality_metric_in_initial_state(
                            simulator, self.quality_metric
                        )
                        states = [initial_state]
                        for act in plan.actions:
                            vprint(f"\taction: {act}")
                            state_test = simulator.apply(initial_state, act)
                            states.append(state_test)
                            vprint(
                                f"\tgens new value: {[float(state_test.get_value(self.pgen_exp[g][0]).constant_value()) for g in range(self.nb_gens)]}"
                            )
                            vprint(
                                f"\tflows new value: {[float(state_test.get_value(self.flows_exp[k][0]).constant_value()) for k in range(self.nb_transmission_lines)]}"
                            )
                            vprint(
                                f"\tcongestions new value: {[state_test.get_value(self.congestions_exp[k][0]) for k in range(self.nb_transmission_lines)]}"
                            )
                            vprint(
                                f"\tupdate status new value: {[state_test.get_value(self.update_status_exp[k][0]) for k in range(self.nb_transmission_lines)]}"
                            )
                            vprint(
                                f"\tgen slack new value: {float(state_test.get_value(self.pgen_exp[1][0]).constant_value())}"
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
                            vprint(f"\tcost: {float(minimize_cost_value)}")
        return plan.actions
