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
from plan4grid.utils import verbose_print


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
        # TODO: refactor this, not clean but not a priority

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
                cfg.CONNECTED_STATUS
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
        tactical_horizon: int,
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
        self.scenario_id = scenario_id
        self.env.set_id(self.scenario_id)
        self.initial_topology = deepcopy(self.env).reset().connectivity_matrix()
        self.curr_obs = self.env.reset()
        self.tactical_horizon = tactical_horizon
        self.static_properties = self.get_static_properties()
        self.mutable_properties = self.get_mutable_properties()
        self.ptdf = self.get_ptdf()
        self.time_step = self.curr_obs.delta_time  # time step in minutes
        self.solver = solver
        self.debug = debug

    def print_summary(self):
        """Print the parameters of the agent."""
        print("Parameters of the agent:")
        print(f"\tTactical horizon: {self.tactical_horizon}")
        print(f"\tTime step: {self.time_step} minutes")
        print(f"\tSolver: {self.solver}")
        print(f"\tDebug mode: {self.debug}\n")

    def print_grid_properties(self):
        """Print the properties of the grid."""
        grid_properties = {**self.static_properties, **self.mutable_properties}
        print("Properties of the grid:")
        print(f"\tGenerators:")
        for key, value in grid_properties[cfg.GENERATORS].items():
            print(f"\t\t{key}: {value}")
        print(f"\tStorages:")
        for key, value in grid_properties[cfg.STORAGES].items():
            print(f"\t\t{key}: {value}")
        print(f"\tTransmission lines:")
        for key, value in grid_properties[cfg.TRANSMISSION_LINES].items():
            print(f"\t\t{key}: {value}")
        print()

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

        simulated_observations = [
            self.curr_obs.simulate(do_nothing_action, i)[0]
            for i in range(self.tactical_horizon)
        ]

        forecasted_states = [
            {
                cfg.GEN_PROD: simulated_observations[t].gen_p,
                cfg.LOADS: simulated_observations[t].load_p,
                cfg.STO_CHARGE: simulated_observations[t].storage_charge,
                cfg.FLOWS: simulated_observations[t].p_or,
                cfg.CONGESTED_STATUS: simulated_observations[t].rho >= 1,
            }
            for t in range(self.tactical_horizon)
        ]

        initial_states = {
            cfg.GEN_PROD: self.curr_obs.gen_p,
            cfg.LOADS: self.curr_obs.load_p,
            cfg.STO_CHARGE: self.curr_obs.storage_charge,
            cfg.FLOWS: self.curr_obs.p_or,
            cfg.CONGESTED_STATUS: self.curr_obs.rho >= 1,
        }

        return initial_states, forecasted_states

    def check_congestions(self, verbose: bool = True) -> bool:
        """This function checks if there is a congestion on the grid on the current observation and on the forecasted observations.

        Args:
            verbose (bool, optional): If True, print information about the congestion. Defaults to True.

        Returns:
            bool: True if there is a congestion, False otherwise
        """
        vprint = verbose_print(verbose)

        congested_now = np.any(self.curr_obs.rho >= 1)
        forecasted_congestions = [
            self.forecasted_states[t][cfg.CONGESTED_STATUS]
            for t in range(self.tactical_horizon)
        ]
        congested_future = np.any(forecasted_congestions)

        def _print_congested_line(line, flow, max_flow, time_step):
            vprint(
                f"\t\tLine {line} is congested with a flow of {flow:.2f} MW,",
                f"but have a maximum/minimum flow of +/- {max_flow:.2f} MW",
                time_step,
                "\n",
            )

        if congested_now:
            vprint("\tCongestion detected!")
            congested_lines = np.where(self.curr_obs.rho >= 1)[0]
            for line in congested_lines:
                max_flow = self.mutable_properties[cfg.TRANSMISSION_LINES][line][
                    cfg.MAX_FLOW
                ]
                flow = self.curr_obs.p_or[line]
                _print_congested_line(line, flow, max_flow, "right now")
            return True

        if congested_future and self.tactical_horizon > 1:
            vprint("\tCongestion detected in the future!")
            first_congestion_at = np.where(forecasted_congestions)[0][0]
            congested_lines = np.where(
                self.forecasted_states[first_congestion_at][cfg.CONGESTED_STATUS]
            )[0]
            for line in congested_lines:
                max_flow = self.mutable_properties[cfg.TRANSMISSION_LINES][line][
                    cfg.MAX_FLOW
                ]
                forecasted_flow = self.forecasted_states[first_congestion_at][
                    cfg.FLOWS
                ][line]
                _print_congested_line(
                    line,
                    forecasted_flow,
                    max_flow,
                    f"in {first_congestion_at+1} time steps",
                )
            return True

        return False

    def check_topology(self, verbose: bool = True) -> bool:
        """This function checks if the topology of the grid has changed on the current observation.

        Args:
            verbose (bool, optional): If True, print information about the congestion. Defaults to True.

        Returns:
            bool: True if the topology has changed, False otherwise
        """
        vprint = verbose_print(verbose)
        current_topology = self.curr_obs.connectivity_matrix()
        topology_unchanged = np.array_equal(self.initial_topology, current_topology)
        if not topology_unchanged:
            vprint("\tTopology has changed!")
            disconnected_lines = np.where(self.initial_topology & ~current_topology)[0]
            connected_lines = np.where(~self.initial_topology & current_topology)[0]
            for line in disconnected_lines:
                vprint(f"\t\tLine {line} has been disconnected.")
            for line in connected_lines:
                vprint(f"\t\tLine {line} has been connected.")
            vprint("\tUpdating PTDF matrix and mutable properties...")
            self.ptdf = self.get_ptdf()
            self.mutable_properties = self.get_mutable_properties()
            self.initial_topology = current_topology
            vprint("\tDone!")
        return not topology_unchanged

    def update_states(self):
        """This function updates the initial states and the forecasted states of the grid."""
        self.initial_states, self.forecasted_states = self.get_states()

    def up_actions_to_g2op_actions(
        self, up_actions: list[InstantaneousAction]
    ) -> list[dict[str, list[tuple[int, float]]]]:
        """This function converts the actions of the UP problem to the actions of the grid2op environment.

        Args:
            up_actions (list[InstantaneousAction]): list of actions of the UP problem, obtained by solving it with the `UnifiedPlanningProblem` class

        Raises:
            RuntimeError: if the action type is not valid, it should be either `prod_target` or `storage_target`s

        Returns:
            list[dict[str, list[tuple[int, float]]]]: transposed actions of the UP problem to the `grid2op` environment, one action dict per time step
        """
        # first we create the list of dict that will be returned
        template_dict = {cfg.REDISPATCH: [], cfg.SET_STORAGE: []}
        g2op_actions = [deepcopy(template_dict)]
        # fetch the slack bus id
        slack_id = np.where(self.static_properties[cfg.GENERATORS][cfg.SLACK])[0][0]
        # then we parse the up actions
        for action in up_actions:
            action = action.action.name
            if action == cfg.ADVANCE_STEP_ACTION:
                new_dict = deepcopy(template_dict)
                g2op_actions.append(new_dict)
                continue
            # we split the string to extract the information
            action_info = action.split("_")
            action_type = action_info[0]
            id = int(action_info[1])
            direction = action_info[2]
            if direction == "increase":
                value = float(action_info[3])
            elif direction == "decrease":
                value = -float(action_info[3])
            else:
                raise RuntimeError(
                    f"The direction of the action is not valid, it should be either increase or decrease!"
                )
            # we get the current value of the generator or the storage
            if action_type == cfg.GENERATOR_ACTION_PREFIX:
                g2op_actions[-1][cfg.REDISPATCH].append((id, value))
            elif action_type == cfg.STORAGE_ACTION_PREFIX:
                g2op_actions[-1][cfg.SET_STORAGE].append((id, value))
            else:
                raise RuntimeError("The action type is not valid!")
            g2op_actions[-1][cfg.REDISPATCH].append((slack_id, -value))
        return g2op_actions

    def get_UP_actions(self, step: int, verbose: bool = True) -> list[ActionSpace]:
        """This function returns the `UnifiedPlanning` actions to perform on the grid.

        Args:
            step (int): current step of the simulation
            verbose (bool, optional): If True, print information about the congestion. Defaults to True.

        Returns:
            list[ActionSpace]: grid2op actions to perform on the grid (one `ActionSpace` per time step)
        """
        vprint = verbose_print(verbose)
        vprint("\tCreating UP problem...")
        grid_params = {**self.static_properties, **self.mutable_properties}
        upp = UnifiedPlanningProblem(
            self.tactical_horizon,
            self.time_step,
            self.ptdf,
            grid_params,
            self.initial_states,
            self.forecasted_states,
            self.solver,
            problem_id=step,
            debug=self.debug,
        )
        if self.debug:
            vprint(f"\tSaving UP problem in {cfg.LOG_DIR}")
            upp.save_problem()
        vprint("\tSolving UP problem...")
        start = timer()
        up_plan = upp.solve()
        end = timer()
        vprint(f"\tProblem solved in {end - start} seconds")
        g2op_actions = [
            self.env.action_space(d) for d in self.up_actions_to_g2op_actions(up_plan)
        ]
        if len(g2op_actions) != self.tactical_horizon:
            # extend the actions to the tactical horizon with do nothing actions
            g2op_actions.extend(
                [
                    self.env.action_space({})
                    for _ in range(self.tactical_horizon - len(g2op_actions))
                ]
            )
        return g2op_actions

    def progress(self, step: int) -> tuple[BaseObservation, float, bool, dict]:
        """This function performs one step of the simulation.

        Args:
            step (int): current step of the simulation

        Returns:
            tuple[BaseObservation, float, bool, dict]: respectively the observation, the reward, the done flag and the info dict
        """
        results = []
        self.update_states()
        if self.check_congestions() or self.check_topology():
            actions = self.get_UP_actions(step)
        else:
            print("\tNo congestion detected, doing nothing...")
            actions = [self.env.action_space({}) for _ in range(self.tactical_horizon)]
        i = 0
        while i != self.tactical_horizon:
            all_zeros = not actions[i].to_vect().any()
            if not all_zeros:
                print(f"\tPerforming action {actions[i]}")
            obs, reward, done, info = self.env.step(actions[i])
            results.append((obs, reward, done, info))
            self.curr_obs = results[-1][0]
            self.update_states()
            if all_zeros and (
                self.check_congestions(verbose=False)
                or self.check_topology(verbose=False)
            ):
                print(
                    "\n\tDoing nothing induced a change in the grid topology or the congestion state, re-solving the UP problem...\n"
                )
                actions = [None in range(i + 1)] + self.get_UP_actions(step)
            i += 1
        return results[-1]
