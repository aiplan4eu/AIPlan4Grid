import warnings
from copy import deepcopy
from logging import DEBUG, INFO
from math import atan, cos, sqrt
from os.path import join as pjoin
from time import perf_counter
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
from grid2op.Action import ActionSpace
from grid2op.Environment import Environment
from grid2op.Episode import EpisodeData
from grid2op.PlotGrid import PlotMatplot
from grid2op.Runner.aux_fun import _aux_add_data
from pandapower.pd2ppc import _pd2ppc
from pandapower.pypower.makePTDF import makePTDF

import plan4grid.config as cfg
from plan4grid.UnifiedPlanningProblem import InstantaneousAction, UnifiedPlanningProblem
from plan4grid.utils import setup_logger


class AIPlan4GridAgent:
    """
    This class implements the AIPlan4Grid agent
    """

    def _create_episode(self, path_save: str = cfg.AGENT_DIR) -> EpisodeData:
        """This function creates the episode data `g2op` object that will be used to store the results.

        Args:
            path_save (str, optional): path where to save the results. Defaults to `cfg.RESULTS_DIR`.

        Returns:
            EpisodeData: episode data object
        """
        nb_timestep_max = self.env.chronics_handler.max_timestep()
        efficient_storing = nb_timestep_max > 0
        nb_timestep_max = max(nb_timestep_max, 0)

        disc_lines_templ = np.full((1, self.env.backend.n_line), fill_value=False, dtype=np.bool_)
        attack_templ = np.full((1, self.env._oppSpace.action_space.size()), fill_value=0.0, dtype=np.float32)

        if efficient_storing:
            times = np.full(nb_timestep_max, fill_value=np.NaN, dtype=np.float32)
            rewards = np.full(nb_timestep_max, fill_value=np.NaN, dtype=np.float32)
            actions = np.full((nb_timestep_max, self.env.action_space.n), fill_value=np.NaN, dtype=np.float32)
            env_actions = np.full(
                (nb_timestep_max, self.env._helper_action_env.n),
                fill_value=np.NaN,
                dtype=np.float32,
            )
            observations = np.full(
                (nb_timestep_max + 1, self.env.observation_space.n),
                fill_value=np.NaN,
                dtype=np.float32,
            )
            disc_lines = np.full((nb_timestep_max, self.env.backend.n_line), fill_value=np.NaN, dtype=np.bool_)
            attack = np.full(
                (nb_timestep_max, self.env._opponent_action_space.n),
                fill_value=0.0,
                dtype=np.float32,
            )
            legal = np.full(nb_timestep_max, fill_value=True, dtype=np.bool_)
            ambiguous = np.full(nb_timestep_max, fill_value=False, dtype=np.bool_)
        else:
            times = np.full(0, fill_value=np.NaN, dtype=np.float32)
            rewards = np.full(0, fill_value=np.NaN, dtype=np.float32)
            actions = np.full((0, self.env.action_space.n), fill_value=np.NaN, dtype=np.float32)
            env_actions = np.full((0, self.env._helper_action_env.n), fill_value=np.NaN, dtype=np.float32)
            observations = np.full((0, self.env.observation_space.n), fill_value=np.NaN, dtype=np.float32)
            disc_lines = np.full((0, self.env.backend.n_line), fill_value=np.NaN, dtype=np.bool_)
            attack = np.full((0, self.env._opponent_action_space.n), fill_value=0.0, dtype=np.float32)
            legal = np.full(0, fill_value=True, dtype=np.bool_)
            ambiguous = np.full(0, fill_value=False, dtype=np.bool_)

        if efficient_storing:
            observations[0, :] = self.env.reset().to_vect()
        else:
            observations = np.concatenate((observations, self.env.reset().to_vect().reshape(1, -1)))

        episode = EpisodeData(
            actions=actions,
            env_actions=env_actions,
            observations=observations,
            rewards=rewards,
            disc_lines=disc_lines,
            times=times,
            observation_space=self.env.observation_space,
            action_space=self.env.action_space,
            helper_action_env=self.env._helper_action_env,
            path_save=path_save,
            disc_lines_templ=disc_lines_templ,
            attack_templ=attack_templ,
            attack=attack,
            attack_space=self.env._opponent_action_space,
            name=self.env.chronics_handler.get_name(),
            force_detail=True,
            other_rewards=[],
            legal=legal,
            ambiguous=ambiguous,
            has_legal_ambiguous=True,
            logger=setup_logger("EpisodeData"),
        )

        episode.observations.objects[0] = episode.observations.helper.from_vect(observations[0, :])
        episode.set_parameters(self.env)
        return episode

    def get_grid_properties(self) -> dict:
        """This function returns the properties of the grid.

        Raises:
            RuntimeError: if the slack bus is not unique

        Returns:
            dict: properties of the grid
        """
        grid_properties = {cfg.GENERATORS: {}, cfg.STORAGES: {}, cfg.TRANSMISSION_LINES: {}}

        # Generators parameters
        grid_properties[cfg.GENERATORS][cfg.PMIN] = self.env.gen_pmin
        grid_properties[cfg.GENERATORS][cfg.PMAX] = self.env.gen_pmax
        grid_properties[cfg.GENERATORS][cfg.REDISPATCHABLE] = self.env.gen_redispatchable
        grid_properties[cfg.GENERATORS][cfg.MAX_RAMP_UP] = self.env.gen_max_ramp_up
        grid_properties[cfg.GENERATORS][cfg.MAX_RAMP_DOWN] = self.env.gen_max_ramp_down
        grid_properties[cfg.GENERATORS][cfg.GEN_COST_PER_MW] = self.env.gen_cost_per_MW
        grid_properties[cfg.GENERATORS][cfg.SLACK] = self.env.backend._grid.gen[cfg.SLACK].to_numpy()
        if len(np.where(grid_properties[cfg.GENERATORS][cfg.SLACK] == True)[0]) > 1:
            raise RuntimeError("There should be only one slack bus, but there are several.")
        grid_properties[cfg.GENERATORS][cfg.GEN_TO_SUBID] = self.env.gen_to_subid
        grid_properties[cfg.GENERATORS][cfg.GEN_BUS] = self.env.current_obs.gen_bus

        # Storages parameters
        grid_properties[cfg.STORAGES][cfg.EMAX] = self.env.storage_Emax
        grid_properties[cfg.STORAGES][cfg.EMIN] = self.env.storage_Emin
        grid_properties[cfg.STORAGES][cfg.LOSS] = self.env.storage_loss
        grid_properties[cfg.STORAGES][cfg.CHARGING_EFFICIENCY] = self.env.storage_charging_efficiency
        grid_properties[cfg.STORAGES][cfg.DISCHARGING_EFFICIENCY] = self.env.storage_discharging_efficiency
        grid_properties[cfg.STORAGES][cfg.STORAGE_MAX_P_PROD] = self.env.storage_max_p_prod
        grid_properties[cfg.STORAGES][cfg.STORAGE_MAX_P_ABSORB] = self.env.storage_max_p_absorb
        grid_properties[cfg.STORAGES][cfg.STORAGE_COST_PER_MW] = self.env.storage_marginal_cost
        grid_properties[cfg.STORAGES][cfg.STORAGE_TO_SUBID] = self.env.storage_to_subid
        grid_properties[cfg.STORAGES][cfg.STORAGE_BUS] = self.env.current_obs.storage_bus

        # Transmission lines parameters
        max_flows = np.array(
            (
                (self.env.backend.lines_or_pu_to_kv * self.env.current_obs.thermal_limit / 1000)
                * sqrt(3)  # triple phase
                * cos(atan(0.4))
            ),
            dtype=float,
        )  # from Ampere to MW

        nb_lines = self.env.current_obs.n_line
        or_idx = self.env.current_obs.line_or_to_subid
        ex_idx = self.env.current_obs.line_ex_to_subid

        for i in range(nb_lines):
            grid_properties[cfg.TRANSMISSION_LINES][i] = {
                cfg.FROM_BUS: or_idx[i],
                cfg.TO_BUS: ex_idx[i],
            }

        for i in grid_properties[cfg.TRANSMISSION_LINES].keys():
            grid_properties[cfg.TRANSMISSION_LINES][i][cfg.CONNECTED_STATUS] = self.env.current_obs.line_status[i]
            grid_properties[cfg.TRANSMISSION_LINES][i][cfg.MAX_FLOW] = max_flows[i]

        return grid_properties

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
        scenario_id: Union[int, str],
        tactical_horizon: int,
        solver: str,
        test: bool,
        debug: bool,
        _nb_gen_actions: int,
        _nb_sto_actions: int,
    ):
        # WARNING: THE ORDER OF THE FOLLOWING SETS IS IMPORTANT
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
        self.test = test
        if self.test and isinstance(scenario_id, str):
            raise ValueError(
                "The scenario ID cannot be a string in test mode because you don't have access to all the chronics."
            )
        self.scenario_id = scenario_id
        self.env.set_id(self.scenario_id)
        self.env.reset()
        self.initial_topology = deepcopy(self.env).reset().connectivity_matrix().astype(bool)
        self.tactical_horizon = tactical_horizon
        self.grid_properties = self.get_grid_properties()
        self.ptdf = self.get_ptdf()
        self.time_step = self.env.current_obs.delta_time  # time step in minutes
        self.solver = solver
        self.debug = debug
        self._nb_gen_actions = _nb_gen_actions
        self._nb_sto_actions = _nb_sto_actions

        if self.debug:
            level = DEBUG
        else:
            level = INFO

        name = __name__.split(".")[-1]
        self.log_file = pjoin(cfg.LOG_DIR, f"{name}{cfg.LOG_SUFFIX}")
        self.logger = setup_logger(name=name, level=level)

        self.episode = self._create_episode()

    def print_summary(self):
        """Print the parameters of the agent."""
        print("Parameters of the agent:\n")
        print(f"\tTactical horizon: {self.tactical_horizon}")
        print(f"\tTime step: {self.time_step} minutes")
        print(f"\tSolver: {self.solver}")
        print(f"\tDebug mode: {self.debug}")
        print()

    def print_grid_properties(self):
        """Print the properties of the grid."""
        print("Properties of the grid:\n")
        print(f"\tGenerators:\n")
        for key, value in self.grid_properties[cfg.GENERATORS].items():
            print(f"\t\t{key}: {value}")
        print()
        print(f"\tStorages:\n")
        for key, value in self.grid_properties[cfg.STORAGES].items():
            print(f"\t\t{key}: {value}")
        print()
        print(f"\tTransmission lines:\n")
        for key, value in self.grid_properties[cfg.TRANSMISSION_LINES].items():
            print(f"\t\t{key}: {value}")
        print()

    def display_grid(self):
        """Display the current state of the grid."""
        plot_helper = PlotMatplot(self.env.observation_space)
        plot_helper.plot_obs(self.env.current_obs)
        plt.show()

    def get_states(self) -> tuple[dict[str, np.array], list[dict[str, np.array]]]:
        """This function returns the initial states and the forecasted states of the grid.

        Returns:
            tuple[dict[str, np.array], list[dict[str, np.array]]]: initial states and forecasted states
        """
        do_nothing_action = self.env.action_space({})

        simulated_observations = [
            self.env.current_obs.simulate(do_nothing_action, i)[0] for i in range(self.tactical_horizon)
        ]

        forecasted_states = [
            {
                cfg.GEN_PROD: simulated_observations[t].gen_p,
                cfg.LOADS: simulated_observations[t].load_p,
                cfg.STO_CHARGE: simulated_observations[t].storage_charge,
                cfg.FLOWS: simulated_observations[t].p_or,
                cfg.CONGESTED_STATUS: simulated_observations[t].rho >= 1,
                cfg.CONNECTED_STATUS: simulated_observations[t].line_status,
            }
            for t in range(self.tactical_horizon)
        ]

        initial_states = {
            cfg.GEN_PROD: self.env.current_obs.gen_p,
            cfg.LOADS: self.env.current_obs.load_p,
            cfg.STO_CHARGE: self.env.current_obs.storage_charge,
            cfg.FLOWS: self.env.current_obs.p_or,
            cfg.CONGESTED_STATUS: self.env.current_obs.rho >= 1,
            cfg.CONNECTED_STATUS: self.env.current_obs.line_status,
        }

        return initial_states, forecasted_states

    def check_congestions(self, verbose: bool = True) -> bool:
        """This function checks if there is a congestion on the grid on the current observation and on the forecasted
        observations.

        Args:
            verbose (bool, optional): if True, the logger will print the info. Defaults to True.

        Returns:
            bool: True if there is a congestion, False otherwise
        """
        congested_now = np.any(self.env.current_obs.rho >= 1)
        forecasted_congestions = [self.forecasted_states[t][cfg.CONGESTED_STATUS] for t in range(self.tactical_horizon)]
        congested_future = np.any(forecasted_congestions)

        def _log_congested_line(line, flow, max_flow, time_step):
            message = f"\tLine {line} is congested with a flow of {flow:.2f} MW but have a maximum/minimum flow of +/- {max_flow:.2f} MW {time_step}"
            self.logger.info(message)

        if congested_now:
            if verbose:
                self.logger.info("Congestion detected!")
            congested_lines = np.where(self.env.current_obs.rho >= 1)[0]
            for line in congested_lines:
                max_flow = self.grid_properties[cfg.TRANSMISSION_LINES][line][cfg.MAX_FLOW]
                flow = self.env.current_obs.p_or[line]
                if verbose:
                    _log_congested_line(line, flow, max_flow, "right now")
            return True

        if congested_future and self.tactical_horizon > 1:
            if verbose:
                self.logger.info("Congestion detected in the future!")
            first_congestion_at = np.where(forecasted_congestions)[0][0]
            congested_lines = np.where(self.forecasted_states[first_congestion_at][cfg.CONGESTED_STATUS])[0]
            for line in congested_lines:
                max_flow = self.grid_properties[cfg.TRANSMISSION_LINES][line][cfg.MAX_FLOW]
                forecasted_flow = self.forecasted_states[first_congestion_at][cfg.FLOWS][line]
                if verbose:
                    _log_congested_line(
                        line,
                        forecasted_flow,
                        max_flow,
                        f"in {first_congestion_at + 1} time steps",
                    )
            return True

        return False

    def check_topology(self, verbose: bool = True) -> bool:
        """This function checks if the topology of the grid has changed on the current observation.
        If the topology has changed, it updates the PTDF matrix and the properties of the grid.

        Args:
            verbose (bool, optional): if True, the logger will print the info. Defaults to True.

        Returns:
            bool: True if the topology has changed, False otherwise
        """
        current_topology = self.env.current_obs.connectivity_matrix().astype(bool)
        topology_changed = not np.array_equal(self.initial_topology, current_topology)
        if topology_changed:
            if verbose:
                self.logger.info("Topology has changed!")
            disconnected_lines = np.where(self.initial_topology & ~current_topology)[0]
            connected_lines = np.where(~self.initial_topology & current_topology)[0]
            for line in disconnected_lines:
                if verbose:
                    self.logger.info(f"\tLine {line} has been disconnected")
            for line in connected_lines:
                if verbose:
                    self.logger.info(f"\tLine {line} has been connected")
            if verbose:
                self.logger.info("Updating PTDF matrix and grid properties...")
            self.ptdf = self.get_ptdf()
            self.grid_properties = self.get_grid_properties()
            self.initial_topology = current_topology
            if verbose:
                self.logger.info("Done!")
        return topology_changed

    def update_states(self):
        """This function updates the initial states and the forecasted states of the grid."""
        self.initial_states, self.forecasted_states = self.get_states()

    def up_actions_to_g2op_actions(
        self, up_actions: list[InstantaneousAction]
    ) -> list[dict[str, list[tuple[int, float]]]]:
        """This function converts the actions of the UP problem to the actions of the grid2op environment.

        Args: up_actions (list[InstantaneousAction]): list of actions of the UP problem, obtained by solving it with
        the `UnifiedPlanningProblem` class

        Raises:
            RuntimeError: if the action type is not valid, it should be either `prod_target` or `storage_target`s

        Returns: list[dict[str, list[tuple[int, float]]]]: transposed actions of the UP problem to the `grid2op`
        environment, one action dict per time step
        """
        # first we create the list of dict that will be returned
        template_dict = {cfg.REDISPATCH: [], cfg.SET_STORAGE: []}
        g2op_actions = [deepcopy(template_dict)]
        # fetch the slack bus id
        slack_id = np.where(self.grid_properties[cfg.GENERATORS][cfg.SLACK])[0][0]
        # then we parse the up actions
        slack_value = 0
        for action in up_actions:
            action = action.action.name
            if action == cfg.ADVANCE_STEP_ACTION:
                g2op_actions[-1][cfg.REDISPATCH].append((slack_id, slack_value))
                new_dict = deepcopy(template_dict)
                g2op_actions.append(new_dict)
                slack_value = 0
                continue
            # we split the string to extract the information
            action_info = action.split("_")
            action_type = action_info[0]
            id = int(action_info[1])
            direction = action_info[2]
            if direction == cfg.INCREASE_ACTION:
                value = float(action_info[3])
            elif direction == cfg.DECREASE_ACTION:
                value = -float(action_info[3])
            else:
                raise RuntimeError(f"The direction of the action is not valid, it should be in {cfg.DIRECTIONS}!")
            # we get the current value of the generator or the storage
            if action_type == cfg.GENERATOR_ACTION_PREFIX:
                g2op_actions[-1][cfg.REDISPATCH].append((id, value))
                slack_value -= value
            elif action_type == cfg.STORAGE_ACTION_PREFIX:
                g2op_actions[-1][cfg.SET_STORAGE].append((id, value))
                slack_value += value
            else:
                raise RuntimeError("The action type is not valid!")
        return g2op_actions

    def get_UP_actions(self, step: int, verbose: bool = True) -> tuple[list[ActionSpace], float, float]:
        """This function returns the `UnifiedPlanning` actions to perform on the grid and the time it took to solve the associated UP problem.

        Args:
            step (int): current step of the simulation
            verbose (bool, optional): if True, the logger will print the steps of the algorithm. Defaults to True.

        Returns:
            tuple[list[ActionSpace], float, float]: grid2op actions to perform on the grid (one `ActionSpace` per time step),
            the beginning time of the solving of the UP problem and the end time of the solving of the UP problem.
        """
        if verbose:
            self.logger.info("\n")
            self.logger.info(f"Creating UP problem number {step}...")
        upp = UnifiedPlanningProblem(
            tactical_horizon=self.tactical_horizon,
            time_step=self.time_step,
            ptdf=self.ptdf,
            grid_params=self.grid_properties,
            initial_states=self.initial_states,
            forecasted_states=self.forecasted_states,
            solver=self.solver,
            problem_id=step,
            debug=self.debug,
            _nb_gen_actions=self._nb_gen_actions,
            _nb_sto_actions=self._nb_sto_actions,
        )
        if self.debug:
            if verbose:
                self.logger.info(f"Saving UP problem in {cfg.LOG_DIR}")
            upp.save_problem()
        if verbose:
            self.logger.info("Solving UP problem...")
        beg_ = perf_counter()
        up_plan = upp.solve()
        end_ = perf_counter()
        time_act = end_ - beg_
        if verbose:
            self.logger.info(f"Problem solved in {time_act} seconds")
        g2op_actions = [self.env.action_space(d) for d in self.up_actions_to_g2op_actions(up_plan)]
        if len(g2op_actions) != self.tactical_horizon:
            # extend the actions to the tactical horizon with do nothing actions
            g2op_actions.extend([self.env.action_space({}) for _ in range(self.tactical_horizon - len(g2op_actions))])
        return g2op_actions, beg_, end_

    def check_maintenance(self) -> bool:
        """This function checks if there is a maintenance on the grid on the current observation.
        Set the `lines_to_reconnect` attribute to the list of lines to reconnect and the duration of the maintenance.

        Returns:
            bool: True if there is a maintenance, False otherwise
        """
        lines_to_reconnect = []
        if not self.env.current_obs.line_status.all():
            disconnected_lines = np.where(~self.env.current_obs.line_status)[0]
            duration_maintenance = self.env.current_obs.duration_next_maintenance
            for line in disconnected_lines:
                if duration_maintenance[line] > 0:
                    self.logger.info("Maintenance detected!")
                    self.logger.info(f"\tLine {line} is disconnected for {duration_maintenance[line]} time steps")
                    lines_to_reconnect.append((line, duration_maintenance[line]))
        self.lines_to_reconnect = lines_to_reconnect
        if len(lines_to_reconnect) > 0:
            return True
        return False

    def progress(self) -> tuple[float, float, bool]:
        """This function performs one step of the simulation.

        Returns:
            tuple[float, float, bool]: reward, time it took to perform the actions and if the simulation is done
        """
        with warnings.catch_warnings():
            efficient_storing = self.env.chronics_handler.max_timestep() > 0
            global_time_act = 0
            warnings.simplefilter("ignore")
            self.update_states()
            if self.check_congestions() or self.check_topology():
                actions, beg_, end_ = self.get_UP_actions(self.env.nb_time_step)
                global_time_act += end_ - beg_
            else:
                self.logger.info(f"No congestion detected, doing nothing at time step {self.env.nb_time_step}")
                beg_ = perf_counter()
                actions = [self.env.action_space({}) for _ in range(self.tactical_horizon)]
                end_ = perf_counter()
                global_time_act += end_ - beg_
            i = 0
            while i <= self.tactical_horizon - 1:
                if self.check_maintenance():
                    lines_to_reconnect_in_next_action = [
                        line_id for line_id in self.lines_to_reconnect if line_id[1] == 1
                    ]
                    if len(lines_to_reconnect_in_next_action) > 0:
                        actions[i + 1].line_change_status = lines_to_reconnect_in_next_action
                all_zeros = not actions[i].to_vect().any()
                obs, reward, done, info = self.env.step(actions[i])
                opp_attack = self.env._oppSpace.last_attack
                reward = _aux_add_data(
                    reward,
                    self.env,
                    self.episode,
                    efficient_storing,
                    end_,
                    beg_,
                    actions[i],
                    obs,
                    info,
                    self.env.nb_time_step,
                    opp_attack,
                )
                if self.env.done:
                    break
                self.update_states()
                if all_zeros and (self.check_congestions(verbose=False) or self.check_topology(verbose=False)):
                    self.logger.info("New congestion or topology change detected --> re-solving the UP problem...")
                    actions, beg_, end_ = self.get_UP_actions(self.env.nb_time_step, verbose=False)
                    actions = [None in range(i + 1)] + actions
                    global_time_act += end_ - beg_
                i += 1
            return reward, global_time_act, done
