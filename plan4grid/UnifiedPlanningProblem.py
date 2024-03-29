import warnings
from logging import DEBUG, INFO
from os.path import join as pjoin

import numpy as np
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
        tactical_horizon: int,
        time_step: int,
        ptdf: list[list],
        grid_params: dict,
        initial_states: dict,
        forecasted_states: list[dict],
        solver: str,
        problem_id: int,
        debug: bool,
        _nb_gen_actions: int,
        _nb_sto_actions: int,
    ):
        get_environment().credits_stream = None

        self.tactical_horizon = tactical_horizon
        self.time_step = time_step
        self.ptdf = ptdf
        self.grid_params = grid_params
        self.nb_gens = len(grid_params[cfg.GENERATORS][cfg.PMAX])
        self.nb_storages = len(grid_params[cfg.STORAGES][cfg.EMAX])
        self.initial_states = initial_states
        self.forecasted_states = forecasted_states
        self.nb_transmission_lines = len(grid_params[cfg.TRANSMISSION_LINES])
        self.slack_id = np.where(grid_params[cfg.GENERATORS][cfg.SLACK] == True)[0][0]
        self.solver = solver
        self.id = problem_id
        self.debug = debug
        self._nb_gen_actions = _nb_gen_actions
        self._nb_sto_actions = _nb_sto_actions

        self.nb_digits = 6
        self.float_precision = 10**-self.nb_digits
        self.ptdf_threshold = 0.05

        if self.debug:
            level = DEBUG
        else:
            level = INFO

        name = f"problem_{self.id}"
        self.log_dir = pjoin(cfg.LOG_DIR, name)
        self.log_file = pjoin(self.log_dir, f"{name}{cfg.LOG_SUFFIX}")
        self.logger = setup_logger(name=name, log_dir=self.log_dir, level=level)

        self.create_fluents()
        self.create_actions(self._nb_gen_actions, self._nb_sto_actions)
        self.create_problem()

    def create_fluents(self):
        """Create problem 'variables' so called fluents in PDDL."""
        self.pgen = np.array(
            [
                [Fluent(f"pgen_{gen_id}_at_{t}", RealType()) for t in range(self.tactical_horizon)]
                for gen_id in range(self.nb_gens)
            ]
        )  # pgen is the setpoint of the generator in MW

        self.pgen_exp = np.array(
            [[FluentExp(self.pgen[gen_id][t]) for t in range(self.tactical_horizon)] for gen_id in range(self.nb_gens)]
        )  # pgen_exp allows to simulate the plan

        self.psto = np.array(
            [
                [Fluent(f"psto_{sto_id}_at_{t}", RealType()) for t in range(self.tactical_horizon)]
                for sto_id in range(self.nb_storages)
            ]
        )  # psto is the state of charge (soc) of the storage in MWh

        self.psto_exp = np.array(
            [
                [FluentExp(self.psto[sto_id][t]) for t in range(self.tactical_horizon)]
                for sto_id in range(self.nb_storages)
            ]
        )  # psto_exp allows to simulate the plan

        self.congestions = np.array(
            [
                [Fluent(f"congestion_on_{k}_at_{t}", BoolType()) for t in range(self.tactical_horizon)]
                for k in range(self.nb_transmission_lines)
            ]
        )  # congestions is a boolean that indicates if the line is congested or not

        self.congestions_exp = np.array(
            [
                [FluentExp(self.congestions[k][t]) for t in range(self.tactical_horizon)]
                for k in range(self.nb_transmission_lines)
            ]
        )  # congestions_exp allows to simulate the plan

        self.flows = np.array(
            [
                [Fluent(f"flow_on_{k}_at_{t}", RealType()) for t in range(self.tactical_horizon)]
                for k in range(self.nb_transmission_lines)
            ]
        )  # flows is the flow on the line

        self.flows_exp = np.array(
            [
                [FluentExp(self.flows[k][t]) for t in range(self.tactical_horizon)]
                for k in range(self.nb_transmission_lines)
            ]
        )  # flows_exp allows to simulate the plan

        self.curr_step = Fluent(f"curr_step", IntType())  # curr_step is the current time step

        self.curr_step_exp = FluentExp(self.curr_step)  # curr_step_exp allows to simulate the plan

        self.gen_moved = np.array(
            [
                [Fluent(f"moved_gen_{gen_id}_{t}", BoolType()) for t in range(self.tactical_horizon)]
                for gen_id in range(self.nb_gens)
            ]
        )  # gen_moved is a boolean that indicates if the generator has been moved or not

        self.sto_moved = np.array(
            [
                [Fluent(f"moved_sto_{sto_id}_{t}", BoolType()) for t in range(self.tactical_horizon)]
                for sto_id in range(self.nb_storages)
            ]
        )  # sto_moved is a boolean that indicates if the storage has been moved or not

    def is_disconnected(self, t: int, k: int) -> bool:
        """Check if a line k is disconnected at time t.

        Args:
            t (int): time step
            k (int): line id

        Returns:
            bool: True if the line is disconnected, False otherwise
        """
        if t == 0:
            return self.initial_states[cfg.CONNECTED_STATUS][k] == False
        else:
            return self.forecasted_states[t][cfg.CONNECTED_STATUS][k] == False

    def create_advance_step_action(
        self,
    ) -> tuple[InstantaneousAction, dict[str, float]]:
        """Create advance time action.

        Returns:
            tuple[InstantaneousAction,dict[str, float]]: advance time action and its cost
        """
        advance_step_action = InstantaneousAction("advance_step")
        advance_step_action.add_precondition(
            And(
                GE(self.curr_step, 0),
                LE(self.curr_step, self.tactical_horizon - 1),
            )
        )
        advance_step_action.add_increase_effect(self.curr_step, 1)
        return advance_step_action, {advance_step_action: 0}

    def create_gen_actions(
        self,
        direction: str,
        nb_actions: int,
    ) -> tuple[list[InstantaneousAction], dict[str, float]]:
        """Create increase or decrease actions for generators.

        Args:
            direction (str): 'increase' or 'decrease'
            nb_actions (int): number of actions + 1 to create between 0 and ramp

        Returns:
            tuple[list[InstantaneousAction], dict[str, float]]: list of generators actions and their costs
        """
        assert direction in cfg.DIRECTIONS
        actions_costs = {}
        pgen_actions = []
        pmin_slack = float(self.grid_params[cfg.GENERATORS][cfg.PMIN][self.slack_id])
        pmax_slack = float(self.grid_params[cfg.GENERATORS][cfg.PMAX][self.slack_id])

        for t in range(self.tactical_horizon):
            for id in range(self.nb_gens):
                pmin = float(self.grid_params[cfg.GENERATORS][cfg.PMIN][id])
                pmax = float(self.grid_params[cfg.GENERATORS][cfg.PMAX][id])
                if (
                    self.grid_params[cfg.GENERATORS][cfg.REDISPATCHABLE][id] == True
                    and self.grid_params[cfg.GENERATORS][cfg.SLACK][id] == False
                ):
                    connected_bus = int(self.grid_params[cfg.GENERATORS][cfg.GEN_BUS][id]) * int(
                        self.grid_params[cfg.GENERATORS][cfg.GEN_TO_SUBID][id]
                    )

                    if t == 0:
                        curr_state = float(self.initial_states[cfg.GEN_PROD][id])
                    else:
                        curr_state = self.pgen[id][t - 1]

                    if direction == cfg.INCREASE_ACTION:
                        ramp = self.grid_params[cfg.GENERATORS][cfg.MAX_RAMP_UP][id]
                    else:
                        ramp = self.grid_params[cfg.GENERATORS][cfg.MAX_RAMP_DOWN][id]

                    for i in np.linspace(0, ramp, nb_actions + 1):
                        if i == 0:
                            continue

                        pgen_actions.append(
                            InstantaneousAction(f"gen_{id}_{direction}_{i}_{t}")
                        )  # this action represents the increase or decrease of the setpoint of the generator by i MW
                        # at time t
                        action = pgen_actions[-1]
                        nb_lines_effects = 0

                        action.add_precondition(
                            Equals(self.curr_step, t),
                        )

                        action.add_precondition(
                            Equals(self.curr_step, t),
                        )
                        action.add_precondition(
                            Or([self.congestions[k][t] for k in range(self.nb_transmission_lines)]),
                        )

                        if direction == cfg.INCREASE_ACTION:
                            new_setpoint = curr_state + i if t == 0 else Plus(curr_state, i)
                            new_setpoint_slack = Minus(self.pgen[self.slack_id][t], i)
                        else:
                            new_setpoint = curr_state - i if t == 0 else Minus(curr_state, i)
                            new_setpoint_slack = Plus(self.pgen[self.slack_id][t], i)

                        action.add_precondition(
                            And(
                                GE(
                                    new_setpoint,
                                    pmin,
                                ),
                                LE(
                                    new_setpoint,
                                    pmax,
                                ),
                                GE(
                                    new_setpoint_slack,
                                    pmin_slack,
                                ),
                                LE(
                                    new_setpoint_slack,
                                    pmax_slack,
                                ),
                            )
                        )

                        actions_costs[action] = Times(
                            i,
                            float(self.grid_params[cfg.GENERATORS][cfg.GEN_COST_PER_MW][id]),
                        )

                        for t2 in range(t, self.tactical_horizon):
                            action.add_effect(self.pgen[id][t2], new_setpoint)
                            self.logger.debug(f"Creating line effect on gen {id} at time {t2}")

                        for k in range(self.nb_transmission_lines):
                            if self.is_disconnected(t, k):
                                if self.ptdf.shape[0] != self.nb_transmission_lines:
                                    self.ptdf = np.insert(
                                        self.ptdf,
                                        k,
                                        np.zeros(self.ptdf.shape[1]),
                                        axis=0,
                                    )
                                continue

                            diff_flows = (
                                self.ptdf[k][connected_bus]
                                * (new_setpoint - float(self.forecasted_states[t][cfg.GEN_PROD][id]))
                                if t == 0
                                else Times(
                                    self.ptdf[k][connected_bus],
                                    Minus(
                                        new_setpoint,
                                        float(self.forecasted_states[t][cfg.GEN_PROD][id]),
                                    ),
                                )
                            )

                            if t == 0:
                                if (
                                    abs(diff_flows)
                                    <= float(self.grid_params[cfg.TRANSMISSION_LINES][k][cfg.MAX_FLOW])
                                    * self.ptdf_threshold
                                ):
                                    self.logger.debug(
                                        f"Effect of action {action.name} on flow {k} at time {t} is negligible given a precision threshold of {self.ptdf_threshold*100}% of the max flow"
                                    )
                                    continue

                            if abs(self.ptdf[k][connected_bus]) >= self.ptdf_threshold:
                                nb_lines_effects += 1
                                action.add_increase_effect(
                                    self.flows[k][t],
                                    diff_flows,
                                )
                                action.add_effect(
                                    self.congestions[k][t],
                                    True,
                                    condition=Or(
                                        GE(
                                            self.flows[k][t] + diff_flows,
                                            float(self.grid_params[cfg.TRANSMISSION_LINES][k][cfg.MAX_FLOW]),
                                        ),
                                        LE(
                                            self.flows[k][t] + diff_flows,
                                            float(-self.grid_params[cfg.TRANSMISSION_LINES][k][cfg.MAX_FLOW]),
                                        ),
                                    ),
                                )
                                action.add_effect(
                                    self.congestions[k][t],
                                    False,
                                    condition=And(
                                        LT(
                                            self.flows[k][t] + diff_flows,
                                            float(self.grid_params[cfg.TRANSMISSION_LINES][k][cfg.MAX_FLOW]),
                                        ),
                                        GT(
                                            self.flows[k][t] + diff_flows,
                                            float(-self.grid_params[cfg.TRANSMISSION_LINES][k][cfg.MAX_FLOW]),
                                        ),
                                    ),
                                )

                        for t2 in range(t, self.tactical_horizon):
                            action.add_decrease_effect(
                                self.pgen[self.slack_id][t2],
                                new_setpoint - float(self.forecasted_states[t2][cfg.GEN_PROD][id]),
                            )
                            self.logger.debug(f"Creating line effect on slack gen {id} at time {t2}")

                        if nb_lines_effects == 0:
                            actions_costs.popitem()
                            pgen_actions.pop()
                            self.logger.warning(f"Action {action.name} is useless and is removed")
                        else:
                            action.add_effect(self.gen_moved[id][t], True)
                            action.add_precondition(Not(self.gen_moved[id][t]))

        return pgen_actions, actions_costs

    def create_sto_actions(
        self,
        direction: str,
        nb_actions: int,
    ) -> tuple[list[InstantaneousAction], dict[str, float]]:
        """Create actions for storages.

        WE ASSUME THAT THERE IS NO LOSS OF ENERGY IN THE STORAGE BETWEEN TWO TIME STEPS.

        Args: direction (str): 'increase' or 'decrease' nb_actions (int): number of actions + 1 to create between 0
        and ramp (not really a ramp here, just the max charge or discharge of the storage)

        Returns:
            tuple[list[InstantaneousAction], dict[str, float]]: list of storages actions and their costs
        """
        assert direction in cfg.DIRECTIONS
        actions_costs = {}
        psto_actions = []

        pmin_slack = float(self.grid_params[cfg.GENERATORS][cfg.PMIN][self.slack_id])
        pmax_slack = float(self.grid_params[cfg.GENERATORS][cfg.PMAX][self.slack_id])

        # Detect first congestion and allow storage modification only in direction that is opposit to the congestion
        # solving in time step before in order to help for preparation
        first_time_congestion = -1
        line_congested_at_first_congestion = []
        storage_no_charge_before_congestion = []
        storage_no_discharge_before_congestion = []
        for t in range(self.tactical_horizon):
            for k in range(self.nb_transmission_lines):
                if bool(self.forecasted_states[t][cfg.CONGESTED_STATUS][k]):
                    # Congestion detected in time step t on line k
                    first_time_congestion = t
                    line_congested_at_first_congestion.append(k)
                    for id in range(self.nb_storages):
                        connected_bus = int(self.grid_params[cfg.STORAGES][cfg.STORAGE_BUS][id]) * int(
                            self.grid_params[cfg.STORAGES][cfg.STORAGE_TO_SUBID][id]
                        )
                        if self.ptdf[k][connected_bus] > 0:
                            # Do not allow storage id to charge before congestion, since solving congestion will need
                            # storage to charge Unless soc is above 50% of max
                            storage_no_charge_before_congestion.append(id)
                            self.logger.debug(
                                f"Storage {id} adding to no charge before congestion because of first congestion in "
                                f"time step {t} on line {k}"
                            )
                        else:
                            # Do not allow storage id to discharge before congestion, since solving congestion will
                            # need storage to discharge Unless soc is below 50% of max
                            storage_no_discharge_before_congestion.append(id)
                            self.logger.debug(
                                f"Storage {id} adding to no discharge before congestion because of first congestion "
                                f"in time step {t} on line {k}"
                            )
            if line_congested_at_first_congestion and t >= 0:
                break

        for t in range(self.tactical_horizon):
            for id in range(self.nb_storages):
                soc_min = float(self.grid_params[cfg.STORAGES][cfg.EMIN][id])
                soc_max = float(self.grid_params[cfg.STORAGES][cfg.EMAX][id])
                connected_bus = int(self.grid_params[cfg.STORAGES][cfg.STORAGE_BUS][id]) * int(
                    self.grid_params[cfg.STORAGES][cfg.STORAGE_TO_SUBID][id]
                )

                if t == 0:
                    curr_state = float(self.initial_states[cfg.STO_CHARGE][id])
                else:
                    curr_state = self.psto[id][t - 1]

                if direction == cfg.INCREASE_ACTION:
                    ramp = self.grid_params[cfg.STORAGES][cfg.STORAGE_MAX_P_ABSORB][id]
                    efficiency = self.grid_params[cfg.STORAGES][cfg.CHARGING_EFFICIENCY][id]
                else:
                    ramp = self.grid_params[cfg.STORAGES][cfg.STORAGE_MAX_P_PROD][id]
                    efficiency = self.grid_params[cfg.STORAGES][cfg.DISCHARGING_EFFICIENCY][id]

                # Note that here the ramp is also in MW
                for i in np.linspace(0, ramp, nb_actions + 1):
                    if i == 0:
                        continue

                    psto_actions.append(
                        InstantaneousAction(f"sto_{id}_{direction}_{i}_{t}")
                    )  # this action represents the charge or discharge of the storage by i MW at time t
                    action = psto_actions[-1]
                    nb_lines_effects = 0
                    actions_costs[action] = i * float(self.grid_params[cfg.STORAGES][cfg.STORAGE_COST_PER_MW][id])

                    action.add_precondition(
                        Equals(self.curr_step, t),
                    )

                    if id in storage_no_discharge_before_congestion and direction == cfg.DECREASE_ACTION:
                        action.add_precondition(
                            Or(
                                GE(
                                    curr_state,
                                    soc_max / 2,
                                ),
                                GE(self.curr_step, int(first_time_congestion)),
                            )
                        )
                    if id in storage_no_charge_before_congestion and direction == cfg.INCREASE_ACTION:
                        action.add_precondition(
                            Or(
                                LE(
                                    curr_state,
                                    soc_max / 2,
                                ),
                                GE(self.curr_step, int(first_time_congestion)),
                            )
                        )

                    if direction == cfg.INCREASE_ACTION:
                        new_soc = Plus(curr_state, efficiency * i * self.time_step / 60)
                        new_setpoint_slack = Plus(self.pgen[self.slack_id][t], i)
                    else:
                        new_soc = Minus(curr_state, i * self.time_step / (60 * efficiency))
                        new_setpoint_slack = Minus(self.pgen[self.slack_id][t], i)

                    action.add_precondition(
                        And(
                            GE(new_soc, soc_min),
                            LE(new_soc, soc_max),
                            GE(new_setpoint_slack, pmin_slack),
                            LE(new_setpoint_slack, pmax_slack),
                        )
                    )
                    action.add_effect(self.psto[id][t], new_soc)
                    for t2 in range(t + 1, self.tactical_horizon):
                        action.add_effect(self.psto[id][t2], new_soc)
                        self.logger.debug(f"Creating line effect on sto {id} at time {t2}")

                    for k in range(self.nb_transmission_lines):
                        if self.is_disconnected(t, k):
                            if self.ptdf.shape[0] != self.nb_transmission_lines:
                                self.ptdf = np.insert(self.ptdf, k, np.zeros(self.ptdf.shape[1]), axis=0)
                            continue

                        if direction == cfg.INCREASE_ACTION:
                            diff_flows = -self.ptdf[k][connected_bus] * i
                        else:
                            diff_flows = self.ptdf[k][connected_bus] * i

                        if (
                            abs(diff_flows)
                            <= float(self.grid_params[cfg.TRANSMISSION_LINES][k][cfg.MAX_FLOW]) * self.ptdf_threshold
                        ):
                            self.logger.debug(
                                f"Effect of action {action.name} on flow {k} at time {t} is negligible given a "
                                f"precision threshold of {self.ptdf_threshold*100}% of the max flow"
                            )
                            continue

                        action.add_increase_effect(self.flows[k][t], diff_flows)
                        nb_lines_effects += 1
                        action.add_effect(
                            self.congestions[k][t],
                            True,
                            condition=Or(
                                GE(
                                    Plus(self.flows[k][t], diff_flows),
                                    float(self.grid_params[cfg.TRANSMISSION_LINES][k][cfg.MAX_FLOW]),
                                ),
                                LE(
                                    Plus(self.flows[k][t], diff_flows),
                                    float(-self.grid_params[cfg.TRANSMISSION_LINES][k][cfg.MAX_FLOW]),
                                ),
                            ),
                        )
                        action.add_effect(
                            self.congestions[k][t],
                            False,
                            condition=And(
                                LT(
                                    Plus(self.flows[k][t], diff_flows),
                                    float(self.grid_params[cfg.TRANSMISSION_LINES][k][cfg.MAX_FLOW]),
                                ),
                                GT(
                                    Plus(self.flows[k][t], diff_flows),
                                    float(-self.grid_params[cfg.TRANSMISSION_LINES][k][cfg.MAX_FLOW]),
                                ),
                            ),
                        )

                    for t2 in range(t, self.tactical_horizon):
                        if direction == cfg.INCREASE_ACTION:
                            action.add_decrease_effect(
                                self.pgen[self.slack_id][t2],
                                -i,
                            )
                        else:
                            action.add_decrease_effect(
                                self.pgen[self.slack_id][t2],
                                i,
                            )
                        self.logger.debug(f"Creating line effect on slack gen {id} at time {t2}")

                    if nb_lines_effects == 0:
                        actions_costs.popitem()
                        psto_actions.pop()
                        self.logger.warning(f"Action {action.name} is useless and is removed")
                    else:
                        action.add_effect(self.sto_moved[id][t], True)
                        action.add_precondition(Not(self.sto_moved[id][t]))

        return psto_actions, actions_costs

    def update_max_flows(self):
        """Update max flows of the transmission lines according to the forecasted flows."""
        for k in range(self.nb_transmission_lines):
            max_flow = self.grid_params[cfg.TRANSMISSION_LINES][k][cfg.MAX_FLOW]
            for t in range(self.tactical_horizon):
                forecasted_flow = self.forecasted_states[t][cfg.FLOWS][k]
                if not bool(self.forecasted_states[t][cfg.CONGESTED_STATUS][k]) and forecasted_flow > max_flow:
                    self.grid_params[cfg.TRANSMISSION_LINES][k][cfg.MAX_FLOW] = max(forecasted_flow, max_flow)
                    self.logger.warning(
                        f"Max flow updated for line: {k} from value {max_flow} to new value {forecasted_flow} at time {t}"
                    )

    def create_actions(self, nb_gen_actions: int = 1, nb_sto_actions: int = 1):
        """Create actions for the problem.

        Args: nb_gen_actions (int): number of actions to create between 0 and ramp for generators in each direction,
        so 2*(nb_gen_actions+1) actions are created nb_sto_actions (int): number of actions to create between 0 and
        ramp for storages in each direction, so 2*(nb_sto_actions+1) actions are created
        """
        self.update_max_flows()

        gen_increase_actions, gen_increase_actions_costs = self.create_gen_actions(
            cfg.INCREASE_ACTION,
            nb_gen_actions,
        )
        gen_decrease_actions, gen_decrease_actions_costs = self.create_gen_actions(
            cfg.DECREASE_ACTION,
            nb_gen_actions,
        )

        sto_increase_actions, sto_increase_actions_costs = self.create_sto_actions(
            cfg.INCREASE_ACTION,
            nb_sto_actions,
        )
        sto_decrease_actions, sto_decrease_actions_costs = self.create_sto_actions(
            cfg.DECREASE_ACTION,
            nb_sto_actions,
        )

        self.pgen_actions = gen_increase_actions + gen_decrease_actions
        self.psto_actions = sto_increase_actions + sto_decrease_actions

        self.advance_step_action, advance_step_cost = self.create_advance_step_action()

        self.actions_costs = {
            **gen_increase_actions_costs,
            **gen_decrease_actions_costs,
            **sto_increase_actions_costs,
            **sto_decrease_actions_costs,
            **advance_step_cost,
        }

    def create_problem(self):
        """Create the problem to solve."""
        problem = Problem(f"GridStability_{self.id}")

        # add fluents
        for id in range(self.nb_gens):
            for t in range(self.tactical_horizon):
                problem.add_fluent(self.pgen[id][t])

        for id in range(self.nb_storages):
            for t in range(self.tactical_horizon):
                problem.add_fluent(self.psto[id][t])

        for k in range(self.nb_transmission_lines):
            for t in range(self.tactical_horizon):
                problem.add_fluent(self.congestions[k][t])
                problem.add_fluent(self.flows[k][t])

        problem.add_fluent(self.curr_step)

        for id in range(self.nb_gens):
            for t in range(self.tactical_horizon):
                problem.add_fluent(self.gen_moved[id][t])

        for id in range(self.nb_storages):
            for t in range(self.tactical_horizon):
                problem.add_fluent(self.sto_moved[id][t])

        # add actions
        problem.add_actions(self.pgen_actions)
        problem.add_actions(self.psto_actions)
        problem.add_action(self.advance_step_action)

        # add initial states
        for id in range(self.nb_gens):
            for t in range(self.tactical_horizon):
                problem.set_initial_value(
                    self.pgen[id][t],
                    float(self.forecasted_states[t][cfg.GEN_PROD][id]),
                )

        for id in range(self.nb_storages):
            for t in range(self.tactical_horizon):
                problem.set_initial_value(
                    self.psto[id][t],
                    float(self.forecasted_states[t][cfg.STO_CHARGE][id]),
                )

        for k in range(self.nb_transmission_lines):
            for t in range(self.tactical_horizon):
                problem.set_initial_value(
                    self.congestions[k][t],
                    bool(self.forecasted_states[t][cfg.CONGESTED_STATUS][k]),
                )
                problem.set_initial_value(
                    self.flows[k][t],
                    float(
                        round(
                            self.forecasted_states[t][cfg.FLOWS][k],
                            self.nb_digits,
                        )
                    ),
                )

        problem.set_initial_value(self.curr_step, 0)

        for id in range(self.nb_storages):
            for t in range(self.tactical_horizon):
                problem.set_initial_value(self.sto_moved[id][t], False)

        for id in range(self.nb_gens):
            for t in range(self.tactical_horizon):
                problem.set_initial_value(self.gen_moved[id][t], False)

        # add quality metrics for optimization + goal
        self.quality_metric = up.model.metrics.MinimizeActionCosts(self.actions_costs)
        problem.add_quality_metric(self.quality_metric)

        goal_1 = [
            Iff(self.congestions[k][t], False)
            for k in range(self.nb_transmission_lines)
            for t in range(self.tactical_horizon)
        ]  # is it too restrictive? Maybe to update

        goal_2 = [Equals(self.curr_step, self.tactical_horizon)]

        goals = goal_1 + goal_2

        problem.add_goal(And(goals))

        self.problem = problem

    def save_problem(self):
        """Save the problem in .upp and .pddl formats in a log directory."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                upp_file = "problem_" + str(self.id) + cfg.UPP_SUFFIX
                pddl_file = "problem_" + str(self.id) + cfg.PDDL_SUFFIX
                pddl_domain_file = "problem_domain_" + str(self.id) + cfg.PDDL_SUFFIX

                # upp problem, "upp" stands for unified planning problem
                with open(pjoin(self.log_dir, upp_file), "w") as f:
                    f.write(
                        f"number of fluents: {compute_size_array(self.pgen) + compute_size_array(self.psto)+ compute_size_array(self.congestions) + compute_size_array(self.flows) + compute_size_array(self.sto_moved) + compute_size_array(self.gen_moved) + 1}\n"
                    )
                    f.write(f"number of actions: {len(self.pgen_actions) + len(self.psto_actions) + 1}\n")
                    f.write(self.problem.__str__())
                f.close()

                # pddl problem
                pddl_writer = up.io.PDDLWriter(self.problem, True, True)
                pddl_writer.write_problem(pjoin(self.log_dir, pddl_file))
                pddl_writer.write_domain(pjoin(self.log_dir, pddl_domain_file))
            except Exception as e:
                raise Exception(f"Error while saving problem: {e}")

    def solve(self) -> list[InstantaneousAction]:
        """Solve the problem.

        Returns:
            list[InstantaneousAction]: list of actions of the plan
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with OneshotPlanner(
                name=self.solver,
                problem_kind=self.problem.kind,
                optimality_guarantee=PlanGenerationResultStatus.SOLVED_OPTIMALLY,
            ) as planner:
                output = planner.solve(self.problem)
                plan = output.plan
                if plan is None:
                    self.logger.warning("No plan found!")
                    self.logger.debug(output)
                    return []
                else:
                    self.logger.info(f"Plan found!")
                    self.logger.info(f"Status: {output.status}")
                    self.logger.info(f"{plan}\n")
                    if self.debug and len(plan.actions) > 0:
                        self.logger.debug("Simulating plan...")
                        with SequentialSimulator(problem=self.problem) as simulator:
                            initial_state = simulator.get_initial_state()
                            minimize_cost_value = evaluate_quality_metric_in_initial_state(
                                simulator, self.quality_metric
                            )
                            states = [initial_state]
                            for act in plan.actions:
                                self.logger.debug(f"\taction: {act}")
                                new_state = simulator.apply(states[-1], act)
                                states.append(new_state)
                                self.logger.debug(
                                    f"\tgens new value: {[[float(new_state.get_value(self.pgen_exp[g][t]).constant_value()) for g in range(self.nb_gens)] for t in range(self.tactical_horizon)]}"
                                )
                                self.logger.debug(
                                    f"\tstorages new value: {[[float(new_state.get_value(self.psto_exp[s][t]).constant_value()) for s in range(self.nb_storages)] for t in range(self.tactical_horizon)]}"
                                )
                                self.logger.debug(
                                    f"\tflows new value: {[[float(new_state.get_value(self.flows_exp[k][t]).constant_value()) for k in range(self.nb_transmission_lines)] for t in range(self.tactical_horizon)]}"
                                )
                                self.logger.debug(
                                    f"\tcongestions new value: {[[new_state.get_value(self.congestions_exp[k][t]) for k in range(self.nb_transmission_lines)] for t in range(self.tactical_horizon)]}"
                                )
                                self.logger.debug(
                                    f"\tgen slack new value: {[float(new_state.get_value(self.pgen_exp[self.slack_id][t]).constant_value()) for t in range(self.tactical_horizon)]}"
                                )
                                self.logger.debug(
                                    f"\tcurrent step new value: {float(new_state.get_value(self.curr_step_exp).constant_value())}"
                                )
                                minimize_cost_value = evaluate_quality_metric(
                                    simulator,
                                    self.quality_metric,
                                    minimize_cost_value,
                                    initial_state,
                                    act.action,
                                    act.actual_parameters,
                                    new_state,
                                )
                                self.logger.debug(f"\tcost: {float(minimize_cost_value)}\n")
                    for fh in self.logger.handlers:
                        fh.close()
                    return plan.actions
