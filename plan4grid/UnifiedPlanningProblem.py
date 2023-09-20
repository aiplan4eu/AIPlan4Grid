import os
from os.path import join as pjoin

import numpy as np
import unified_planning as up
from grid2op.Observation import BaseObservation
from unified_planning.engines import PlanGenerationResultStatus
from unified_planning.engines.sequential_simulator import (
    evaluate_quality_metric,
    evaluate_quality_metric_in_initial_state,
)
from unified_planning.shortcuts import *

import plan4grid.config as cfg
from plan4grid.utils import compute_size_array, setup_logger


class UnifiedPlanningProblem:
    """Unified planning problem class that modelled the grid stability problem as a planning problem."""

    def __init__(
        self,
        operational_horizon: int,
        discretization: int,
        ptdf: list[list],
        grid_params: dict,
        initial_states: dict,
        forecasted_states: dict,
        solver: str,
        obs: BaseObservation,
        problem_id: int,
    ):
        get_environment().credits_stream = None

        self.operational_horizon = operational_horizon
        self.discretization = discretization
        self.ptdf = ptdf
        self.grid_params = grid_params
        self.nb_gens = len(grid_params[cfg.GENERATORS][cfg.PMAX])
        self.nb_storages = len(grid_params[cfg.STORAGES][cfg.EMAX])
        self.initial_states = initial_states
        self.forecasted_states = forecasted_states
        self.nb_transmission_lines = len(grid_params[cfg.TRANSMISSION_LINES])
        self.slack_gens = np.where(grid_params[cfg.GENERATORS][cfg.SLACK] == True)[0]
        self.solver = solver
        self.obs = obs
        self.id = problem_id

        self.nb_digits = 6
        self.float_precision = 10**-self.nb_digits

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
        """Create problem 'variables' so called fluents in PDDL."""
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

        self.pstor = np.array(
            [
                [
                    Fluent(f"pstor_{stor_id}_{t}", RealType())
                    for t in range(self.operational_horizon)
                ]
                for stor_id in range(self.nb_storages)
            ]
        )

        self.pstor_exp = np.array(
            [
                [
                    FluentExp(self.pgen[stor_id][t])
                    for t in range(self.operational_horizon)
                ]
                for stor_id in range(self.nb_storages)
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

    def create_gen_actions(self) -> dict[str, float]:
        """Create actions for generators.

        Returns:
            dict[str, float]: dictionary of generators actions and their costs
        """
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

                for i in range(pmin, pmax + 1, self.discretization):
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
                                self.nb_digits,
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

    def create_stor_actions(self) -> dict[str, float]:
        """Create actions for storages.

        Returns:
            dict[str, float]: dictionary of storages actions and their costs
        """
        self.pstor_actions = []
        actions_costs = {}

        for stor_id in range(self.nb_storages):
            soc_max = int(self.grid_params[cfg.STORAGES][cfg.EMAX][stor_id])
            soc_min = int(self.grid_params[cfg.STORAGES][cfg.EMIN][stor_id])
            pmax_charge = self.grid_params[cfg.STORAGES][cfg.STORAGE_MAX_P_PROD][
                stor_id
            ]
            pmax_discharge = self.grid_params[cfg.STORAGES][cfg.STORAGE_MAX_P_ABSORB][
                stor_id
            ]
            charging_efficiency = self.grid_params[cfg.STORAGES][
                cfg.CHARGING_EFFICIENCY
            ][stor_id]
            discharging_efficiency = self.grid_params[cfg.STORAGES][
                cfg.DISCHARGING_EFFICIENCY
            ][stor_id]
            connected_bus = int(self.obs.storage_bus[stor_id]) * int(
                self.grid_params[cfg.STORAGES][cfg.STORAGE_TO_SUBID][stor_id]
            )

            if charging_efficiency == 0 or discharging_efficiency == 0:
                self.logger.warning(
                    f"Storage: {stor_id} has 0 charge or discharge efficiency, no action created!"
                )
                continue
            for i in range(soc_min, soc_max + 1):
                if not (
                    # the max charge/discharge power in grid2Op are given from the grid referencial not from the storage asset.
                    # TODO get time step size from grid2Op
                    self.initial_states[cfg.STORAGES][stor_id]
                    >= i - pmax_charge * 5 / 60 / charging_efficiency
                    and self.initial_states[cfg.STORAGES][stor_id]
                    <= i + pmax_discharge * 5 / 60 * discharging_efficiency
                ):
                    continue

                target_delta_soc = i - self.initial_states[cfg.STORAGES][stor_id]
                if target_delta_soc > 0:
                    target_pcharge = (60 / 5) * target_delta_soc / charging_efficiency
                    target_pdischarge = 0
                elif target_delta_soc < 0:
                    target_pdischarge = (
                        -(60 / 5) * target_delta_soc * discharging_efficiency
                    )
                    target_pcharge = 0
                else:
                    target_pdischarge = 0
                    target_pcharge = 0

                # although there can be a change in the delta SOC forecast (due to loss) we will asusme the the forecasted power charge and dischare are always 0
                # meaning that the forecasted plan for storage is to do nothing.
                self.pstor_actions.append(
                    InstantaneousAction(f"stor_target_{stor_id}_{0}_{i}")
                )
                action = self.pstor_actions[-1]

                actions_costs[action] = float(
                    abs(target_delta_soc)
                    * self.grid_params[cfg.STORAGES][cfg.STORAGE_COST_PER_MW][stor_id]
                )
                action.add_precondition(
                    And(
                        GE(
                            self.pstor[stor_id][0],
                            float(
                                self.forecasted_states[cfg.STORAGES][0][stor_id]
                                - self.float_precision
                            ),
                        ),
                        LE(
                            self.pstor[stor_id][0],
                            float(
                                self.forecasted_states[cfg.STORAGES][0][stor_id]
                                + self.float_precision
                            ),
                        ),
                    )
                )
                action.add_effect(self.pstor[stor_id][0], i)

                for k in range(self.nb_transmission_lines):
                    # not necessary if one time step
                    # diff_charge = float(
                    #    round(
                    #        self.ptdf[k][connected_bus]*target_pcharge,
                    #        self.nb_digits,
                    #        )
                    # )
                    # diff_discharge = float(
                    #    round(
                    #        -self.ptdf[k][connected_bus]*target_pdischare,
                    #        self.nb_digits,
                    #        )
                    # )

                    diff_flow = float(
                        round(
                            self.ptdf[k][connected_bus]
                            * (-target_pcharge + target_pdischarge),
                            self.nb_digits,
                        )
                    )
                    action.add_increase_effect(self.flows[k][0], diff_flow)
                    # action.add_increase_effect(self.flows[k][0],diff_discharge,deltaSOC_withforecasted<0) not necessay if one time step

                    action.add_effect(
                        self.congestions[k][0],
                        True,
                        condition=Or(
                            GE(
                                self.flows[k][0] + diff_flow,
                                float(
                                    self.grid_params[cfg.TRANSMISSION_LINES][k][
                                        cfg.MAX_FLOW
                                    ]
                                ),
                            ),
                            LE(
                                self.flows[k][0] + diff_flow,
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
                                self.flows[k][0] + diff_flow,
                                float(
                                    self.grid_params[cfg.TRANSMISSION_LINES][k][
                                        cfg.MAX_FLOW
                                    ]
                                ),
                            ),
                            GT(
                                self.flows[k][0] + diff_flow,
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
                        -target_pcharge + target_pdischarge,
                    )
        return actions_costs

    def create_actions(self):
        """Create actions for the problem."""
        gen_costs = self.create_gen_actions()
        stor_costs = self.create_stor_actions()

        self.actions_costs = {**gen_costs, **stor_costs}

    def create_problem(self):
        """Create the problem to solve."""
        problem = Problem(f"GridStability_{self.id}")

        # add fluents
        for gen_id in range(self.nb_gens):
            for t in range(self.operational_horizon):
                problem.add_fluent(self.pgen[gen_id][t])

        for stor_id in range(self.nb_storages):
            for t in range(self.operational_horizon):
                problem.add_fluent(self.pstor[stor_id][t])

        for k in range(self.nb_transmission_lines):
            for t in range(self.operational_horizon):
                problem.add_fluent(self.congestions[k][t])
                problem.add_fluent(self.flows[k][t])

        # add actions
        problem.add_actions(self.pgen_actions)
        problem.add_actions(self.pstor_actions)

        # add initial states
        for gen_id in range(self.nb_gens):
            for t in range(self.operational_horizon):
                problem.set_initial_value(
                    self.pgen[gen_id][t],
                    float(self.forecasted_states[cfg.GENERATORS][t][gen_id]),
                )

        for stor_id in range(self.nb_storages):
            for t in range(self.operational_horizon):
                problem.set_initial_value(
                    self.pstor[stor_id][t],
                    float(self.forecasted_states[cfg.STORAGES][t][stor_id]),
                )
                # TODO: correct because no forecasted state for storage when several time step

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
                            self.nb_digits,
                        )
                    ),
                )

        # add quality metrics for optimization + goal
        self.quality_metric = up.model.metrics.MinimizeActionCosts(self.actions_costs)
        problem.add_quality_metric(self.quality_metric)

        goals = [
            Iff(self.congestions[k][t], False)
            for k in range(self.nb_transmission_lines)
            for t in range(self.operational_horizon)
        ]  # is it too restrictive?

        problem.add_goal(And(goals))

        self.problem = problem

    def save_problem(self):
        """Save the problem in .upp and .pddl formats in a temporary directory."""
        upp_file = "problem_" + str(self.id) + cfg.UPP_SUFFIX
        pddl_file = "problem_" + str(self.id) + cfg.PDDL_SUFFIX
        pddl_domain_file = "problem_domain_" + str(self.id) + cfg.PDDL_SUFFIX

        # upp problem, "upp" stands for unified planning problem
        with open(pjoin(self.log_dir, upp_file), "w") as f:
            f.write(
                f"number of fluents: {compute_size_array(self.pgen) + compute_size_array(self.pstor)+ compute_size_array(self.congestions) + compute_size_array(self.flows)}\n"
            )
            f.write(
                f"number of actions: {len(self.pgen_actions) + len(self.pstor_actions)}\n"
            )
            f.write(self.problem.__str__())
        f.close()

        # pddl problem
        pddl_writer = up.io.PDDLWriter(self.problem, True, True)
        pddl_writer.write_problem(pjoin(self.log_dir, pddl_file))
        pddl_writer.write_domain(pjoin(self.log_dir, pddl_domain_file))

    def solve(self, simulate=False) -> list[InstantaneousAction]:
        """Solve the problem.

        Args:
            simulate (bool, optional): If True, simulate the founded plan. Defaults to False.

        Returns:
            list[InstantaneousAction]: list of actions of the plan
        """
        with OneshotPlanner(
            name=self.solver,
            params={"params": "-s WAStar -h hrmax"},
        ) as planner:
            output = planner.solve(self.problem)
            plan = output.plan
            if plan is None:
                print("\tNo plan found!")
                self.logger.warning(output)
                self.logger.warning("Plan returned: []")
                return []
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
                            state_test = simulator.apply(states[-1], act)
                            states.append(state_test)
                            self.logger.debug(
                                f"\tgens new value: {[[float(state_test.get_value(self.pgen_exp[g][t]).constant_value()) for g in range(self.nb_gens)] for t in range(self.operational_horizon)]}"
                            )
                            self.logger.debug(
                                f"\tstorages new value: {[[float(state_test.get_value(self.pstor_exp[s][t]).constant_value()) for s in range(self.nb_storages)] for t in range(self.operational_horizon)]}"
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
