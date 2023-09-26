from copy import deepcopy
from math import atan, cos, sqrt
from timeit import default_timer as timer

import matplotlib.pyplot as plt
import numpy as np
from grid2op.Action import ActionSpace
from grid2op.Environment import Environment
from grid2op.Observation import BaseObservation
from grid2op.PlotGrid import PlotMatplot
from pandapower.pd2ppc import _pd2ppc
from pandapower.pypower.makePTDF import makePTDF

import plan4grid.config as cfg
from plan4grid.UnifiedPlanningProblem import InstantaneousAction, UnifiedPlanningProblem


class AIPlan4GridAgent:
    """
    This class implements the AIPlan4Grid agent
    """

    def get_static_properties(self) -> dict:
        """This function returns the static properties of the grid."""
        static_properties = {cfg.GENERATORS: {}, cfg.STORAGES: {}}

        # Generators parameters
        static_properties[cfg.GENERATORS][cfg.PMIN] = self.env.gen_pmin
        static_properties[cfg.GENERATORS][cfg.PMAX] = self.env.gen_pmax
        static_properties[cfg.GENERATORS][
            cfg.REDISPATCHABLE
        ] = self.env.gen_redispatchable
        static_properties[cfg.GENERATORS][cfg.MAX_RAMP_UP] = self.env.gen_max_ramp_up
        static_properties[cfg.GENERATORS][
            cfg.MAX_RAMP_DOWN
        ] = self.env.gen_max_ramp_down
        static_properties[cfg.GENERATORS][
            cfg.GEN_COST_PER_MW
        ] = self.env.gen_cost_per_MW
        static_properties[cfg.GENERATORS][cfg.SLACK] = self.env.backend._grid.gen[
            cfg.SLACK
        ].to_numpy()
        if len(np.where(static_properties[cfg.GENERATORS][cfg.SLACK] == True)[0]) > 1:
            raise RuntimeError(
                "There should be only one slack bus, but there are several."
            )
        static_properties[cfg.GENERATORS][cfg.GEN_TO_SUBID] = self.env.gen_to_subid

        # Storages parameters
        static_properties[cfg.STORAGES][cfg.EMAX] = self.env.storage_Emax
        static_properties[cfg.STORAGES][cfg.EMIN] = self.env.storage_Emin
        static_properties[cfg.STORAGES][cfg.LOSS] = self.env.storage_loss
        static_properties[cfg.STORAGES][
            cfg.CHARGING_EFFICIENCY
        ] = self.env.storage_charging_efficiency
        static_properties[cfg.STORAGES][
            cfg.DISCHARGING_EFFICIENCY
        ] = self.env.storage_discharging_efficiency
        static_properties[cfg.STORAGES][
            cfg.STORAGE_MAX_P_PROD
        ] = self.env.storage_max_p_prod
        static_properties[cfg.STORAGES][
            cfg.STORAGE_MAX_P_ABSORB
        ] = self.env.storage_max_p_absorb
        static_properties[cfg.STORAGES][
            cfg.STORAGE_COST_PER_MW
        ] = self.env.storage_marginal_cost
        static_properties[cfg.STORAGES][
            cfg.STORAGE_TO_SUBID
        ] = self.env.storage_to_subid

        return static_properties

    def get_mutable_properties(self):
        """This function returns the mutable properties of the grid which depends on the current observation."""
        mutable_properties = {cfg.TRANSMISSION_LINES: {}}

        # Generators parameters
        self.static_properties[cfg.GENERATORS][cfg.GEN_BUS] = self.curr_obs.gen_bus

        # Storages parameters
        self.static_properties[cfg.STORAGES][
            cfg.STORAGE_BUS
        ] = self.curr_obs.storage_bus

        max_flows = np.array(
            (
                (
                    self.env.backend.lines_or_pu_to_kv
                    * self.curr_obs.thermal_limit
                    / 1000
                )
                * sqrt(3)  # triple phase
                * cos(atan(0.4))
            ),
            dtype=float,
        )  # from Ampere to MW

        nb_lines = self.curr_obs.n_line
        or_idx = self.curr_obs.line_or_to_subid
        ex_idx = self.curr_obs.line_ex_to_subid

        for i in range(nb_lines):
            mutable_properties[cfg.TRANSMISSION_LINES][i] = {
                cfg.FROM_BUS: or_idx[i],
                cfg.TO_BUS: ex_idx[i],
            }

        for tl_id in mutable_properties[cfg.TRANSMISSION_LINES].keys():
            mutable_properties[cfg.TRANSMISSION_LINES][tl_id][
                cfg.STATUS
            ] = self.curr_obs.line_status[tl_id]
            mutable_properties[cfg.TRANSMISSION_LINES][tl_id][cfg.MAX_FLOW] = max_flows[
                tl_id
            ]

        return mutable_properties

    def get_ptdf(self):
        """This function returns the PTDF matrix of the grid

        Returns:
            np.array: PTDF matrix
        """
        net = self.env.backend._grid
        _, ppci = _pd2ppc(net)
        ptdf = makePTDF(ppci["baseMVA"], ppci[cfg.BUS], ppci["branch"])
        return ptdf

    def __init__(
        self,
        env: Environment,
        scenario_id: int,
        operational_horizon: int,
        solver: str,
        debug: bool = False,
    ):
        if env.n_storage > 0 and not env.action_space.supports_type("set_storage"):
            raise RuntimeError(
                "Impossible to create this class with an environment that does not allow "
                "modification of storage units when there are storage units on the grid. "
            )
        if not env.action_space.supports_type("redispatch"):
            raise RuntimeError(
                "This type of agent can only perform actions using storage units, curtailment or"
                "redispatching. It requires at least to be able to do redispatching."
            )

        self.env = env
        self.env.set_id(scenario_id)
        self.initial_topology = deepcopy(self.env).reset().connectivity_matrix()
        self.curr_obs = self.env.reset()
        self.operational_horizon = operational_horizon
        self.static_properties = self.get_static_properties()
        self.mutable_properties = self.get_mutable_properties()
        self.ptdf = self.get_ptdf()
        self.time_step = self.curr_obs.delta_time  # time step in minutes
        self.solver = solver
        self.debug = debug

    def display_grid(self):
        """Display the current state of the grid."""

        plot_helper = PlotMatplot(self.env.observation_space)
        plot_helper.plot_obs(self.curr_obs)
        plt.show()

    def get_states(self) -> tuple[dict[str, np.array], dict[str, np.array]]:
        """This function returns the initial states and the forecasted states of the grid.

        Returns:
            tuple[dict[str, np.array], dict[str, np.array]]: initial states and forecasted states
        """
        do_nothing_action = self.env.action_space({})

        simulated_observations = [self.curr_obs.simulate(do_nothing_action)[0]]
        for _ in range(self.operational_horizon - 1):
            obs, *_ = simulated_observations[-1].simulate(do_nothing_action)
            simulated_observations.append(obs)

        forecasted_states = {
            cfg.GENERATORS: np.array(
                [
                    simulated_observations[t].gen_p
                    for t in range(self.operational_horizon)
                ]
            ),
            cfg.LOADS: np.array(
                [
                    simulated_observations[t].load_p
                    for t in range(self.operational_horizon)
                ]
            ),
            cfg.STORAGES: np.array(
                [
                    simulated_observations[t].storage_charge
                    for t in range(self.operational_horizon)
                ]
            ),
            cfg.FLOWS: np.array(
                [
                    simulated_observations[t].p_or
                    for t in range(self.operational_horizon)
                ]
            ),
            cfg.TRANSMISSION_LINES: np.array(
                [
                    simulated_observations[t].rho >= 1
                    for t in range(self.operational_horizon)
                ]
            ),
        }

        initial_states = {
            cfg.GENERATORS: self.curr_obs.gen_p,
            cfg.LOADS: self.curr_obs.load_p,
            cfg.STORAGES: self.curr_obs.storage_charge,
            cfg.FLOWS: self.curr_obs.p_or,
            cfg.TRANSMISSION_LINES: self.curr_obs.rho >= 1,
        }

        return initial_states, forecasted_states

    def check_congestions(self) -> bool:
        """This function checks if there is a congestion on the grid.

        Returns:
            bool: True if there is a congestion, False otherwise
        """
        congested = np.any(self.curr_obs.rho >= 1)
        if congested:
            print("\tCongestion detected!")
            congested_lines = np.where(self.curr_obs.rho >= 1)[0]
            for line in congested_lines:
                print(
                    f"\t\tLine {line} is congested with a flow of {self.curr_obs.p_or[line]:.2f} MW,",
                    f"but have a maximum flow of {self.mutable_properties[cfg.TRANSMISSION_LINES][line][cfg.MAX_FLOW]:.2f} MW",
                )
        return congested

    def check_topology(self) -> bool:
        """This function checks if the topology of the grid has changed.

        Returns:
            bool: True if the topology has changed, False otherwise
        """
        current_topology = self.curr_obs.connectivity_matrix()
        topology_unchanged = np.array_equal(self.initial_topology, current_topology)
        if not topology_unchanged:
            print("\tTopology has changed!")
            disconnected_lines = np.where(self.initial_topology & ~current_topology)[0]
            connected_lines = np.where(~self.initial_topology & current_topology)[0]
            for line in disconnected_lines:
                print(f"\t\tLine {line} has been disconnected.")
            for line in connected_lines:
                print(f"\t\tLine {line} has been connected.")
            print("\tUpdating PTDF matrix and mutable properties...")
            self.ptdf = self.get_ptdf()
            self.mutable_properties = self.get_mutable_properties()
            print("\tDone!")
        return not topology_unchanged

    def update_states(self):
        """This function updates the initial states and the forecasted states of the grid."""
        self.initial_states, self.forecasted_states = self.get_states()

    def up_actions_to_g2op_actions(
        self, up_actions: list[InstantaneousAction]
    ) -> dict[str, list[tuple[int, float]]]:
        """This function converts the actions of the UP problem to the actions of the grid2op environment.

        Args:
            up_actions (list[str]): list of actions of the UP problem, obtained by solving it with the `UnifiedPlanningProblem` class

        Raises:
            RuntimeError: if the action type is not valid, it should be either prod_target or storage_target

        Returns:
            dict[str, list[tuple[int, float]]]: transposed actions of the UP problem to the grid2op environment
        """
        # first we create the dict that will be returned
        g2op_actions = {cfg.REDISPATCH: [], cfg.SET_STORAGE: []}
        slack_actions = []
        # fetch the slack bus id
        slack_id = np.where(self.static_properties[cfg.GENERATORS][cfg.SLACK])[0][0]
        # then we parse the up actions
        for action in up_actions:
            action = action.action.name
            # we split the string to extract the information
            action_info = action.split("_")
            # we get the type of the action
            action_type = action_info[0]
            # we get the id of the generator or the storage
            id = int(action_info[2])
            # we get the time step
            time_step = int(action_info[3])
            # we get the value of the action
            value = float(action_info[4])

            # we get the current value of the generator or the storage
            if action_type == cfg.GENERATOR_ACTION_PREFIX:
                current_value = self.curr_obs.gen_p[id]
            elif action_type == cfg.STORAGE_ACTION_PREFIX:
                current_value = self.curr_obs.storage_charge[id]
            else:
                raise RuntimeError(
                    "The action type is not valid, it should be either prod_target or storage_target"
                )

            # we compute the value of the action
            action_value = value - current_value

            # we add the action to the dict
            if action_type == cfg.GENERATOR_ACTION_PREFIX:
                g2op_actions[cfg.REDISPATCH].append((id, action_value))
            elif action_type == cfg.STORAGE_ACTION_PREFIX:
                g2op_actions[cfg.SET_STORAGE].append((id, action_value))
            else:
                raise RuntimeError(
                    "The action type is not valid, it should be either prod_target or storage_target"
                )
            slack_actions.append((slack_id, -action_value))
        g2op_actions[cfg.REDISPATCH].append(
            (slack_id, sum([v[1] for v in slack_actions]))
        )
        return g2op_actions

    def get_actions(self, step: int) -> ActionSpace:
        """This function returns the actions to perform on the grid.

        Args:
            step (int): current step of the simulation

        Returns:
            ActionSpace: g2op actions to perform on the grid
        """
        self.update_states()
        if self.check_congestions() == False and self.check_topology() == False:
            print(
                "\tNo congestion and no topology change detected, no need to solve UP problem."
            )
            return self.env.action_space({})
        print("\tCreating UP problem...")
        grid_params = {**self.static_properties, **self.mutable_properties}
        upp = UnifiedPlanningProblem(
            self.operational_horizon,
            self.time_step,
            self.ptdf,
            grid_params,
            self.initial_states,
            self.forecasted_states,
            self.solver,
            problem_id=step,
        )
        print(f"\tSaving UP problem in {cfg.LOG_DIR}")
        upp.save_problem()
        print("\tSolving UP problem...")
        start = timer()
        up_plan = upp.solve(simulate=self.debug)
        end = timer()
        print(f"\tProblem solved in {end - start} seconds")
        g2op_actions = self.up_actions_to_g2op_actions(up_plan)
        return self.env.action_space(g2op_actions)

    def progress(self, step: int) -> tuple[BaseObservation, float, bool, dict]:
        """This function performs one step of the simulation.

        Args:
            step (int): current step of the simulation

        Returns:
            tuple[BaseObservation, float, bool, dict]: respectively the observation, the reward, the done flag and the info dict
        """
        actions = self.get_actions(step)
        # print("\tPerforming actions:")
        # print(f"\t{actions}")
        observation, reward, done, info = self.env.step(actions)
        # print(observation.p_or[17])
        self.curr_obs = observation
        return observation, reward, done, info
