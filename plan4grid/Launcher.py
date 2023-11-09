import logging
from time import perf_counter
from typing import Union

import grid2op
from grid2op.Backend import PandaPowerBackend
from grid2op.Chronics import FromHandlers
from grid2op.Chronics.handlers import CSVHandler, DoNothingHandler, NoisyForecastHandler, PerfectForecastHandler
from tqdm import tqdm

import plan4grid.config as cfg
from plan4grid.AIPlan4GridAgent import AIPlan4GridAgent
from plan4grid.utils import clean_agent_results, clean_logs


class Launcher:
    """Launcher class."""

    def _check_parameters(self):
        try:
            if self.parameters[cfg.TACTICAL_HORIZON] > self.parameters[cfg.STRATEGIC_HORIZON]:
                raise ValueError(
                    f"The tactical horizon ({self.parameters[cfg.TACTICAL_HORIZON]}) cannot be greater than the strategic horizon ({self.parameters[cfg.STRATEGIC_HORIZON]})."
                )
            if self.parameters[cfg.TACTICAL_HORIZON] <= 0:
                raise ValueError(
                    f"The tactical horizon ({self.parameters[cfg.TACTICAL_HORIZON]}) must be greater than 0."
                )
            if self.parameters[cfg.STRATEGIC_HORIZON] <= 0:
                raise ValueError(
                    f"The strategic horizon ({self.parameters[cfg.STRATEGIC_HORIZON]}) must be greater than 0."
                )
            if self.parameters[cfg.SOLVER] not in cfg.SOLVERS:
                raise ValueError(
                    f"The solver ({self.parameters[cfg.SOLVER]}) must be one of the following: {cfg.SOLVERS}."
                )
            if self.parameters[cfg.NOISE] not in [True, False]:
                raise ValueError(f"The noise parameter ({self.parameters[cfg.NOISE]}) must be a boolean.")
            if self.parameters[cfg.STRATEGIC_HORIZON] > 288:
                raise ValueError(
                    f"The strategic horizon ({self.parameters[cfg.STRATEGIC_HORIZON]}) must be lower or equal to 288, (24 hours in 5 minutes time steps)."
                )
        except KeyError as e:
            raise KeyError(f"The parameter {e} is missing.")

    def _get_data_feeding_kwargs(self) -> dict:
        if self.noise:
            handler = NoisyForecastHandler
        else:
            handler = PerfectForecastHandler

        return {
            "gridvalueClass": FromHandlers,
            "gen_p_handler": CSVHandler("prod_p"),
            "load_p_handler": CSVHandler("load_p"),
            "gen_v_handler": DoNothingHandler("prod_v"),
            "load_q_handler": CSVHandler("load_q"),
            "h_forecast": [h * self.time_step for h in range(1, self.tactical_horizon + 1)],
            "gen_p_for_handler": handler("prod_p_forecasted"),
            "load_p_for_handler": handler("load_p_forecasted"),
            "load_q_for_handler": handler("load_q_forecasted"),
        }

    def __init__(
        self,
        env_name: str,
        scenario_id: Union[int, str],
        tactical_horizon: int = 1,
        strategic_horizon: int = 288,
        time_step: int = 5,
        solver: str = "enhsp",
        noise: bool = False,
        test: bool = False,
        debug: bool = False,
        _nb_gen_actions: int = 1,
        _nb_sto_actions: int = 1,
    ):
        self.env_name = env_name
        self.test = test
        if self.test and isinstance(scenario_id, str):
            raise ValueError(
                "The scenario ID cannot be a string in test mode because you don't have access to all the chronics."
            )
        self.scenario_id = scenario_id
        self.tactical_horizon = tactical_horizon
        self.strategic_horizon = strategic_horizon
        self.time_step = time_step
        self.solver = solver
        self.noise = noise
        self.debug = debug
        self._nb_gen_actions = _nb_gen_actions
        self._nb_sto_actions = _nb_sto_actions

        self.parameters = {
            cfg.TACTICAL_HORIZON: self.tactical_horizon,
            cfg.STRATEGIC_HORIZON: self.strategic_horizon,
            cfg.SOLVER: self.solver,
            cfg.NOISE: self.noise,
        }
        self._check_parameters()

        clean_logs()
        clean_agent_results()

        try:
            self.env = grid2op.make(
                dataset=self.env_name,
                data_feeding_kwargs=self._get_data_feeding_kwargs(),
                test=self.test,
                backend=PandaPowerBackend(),
            )
        except ValueError as e:
            raise ValueError(f"An error occurred while loading the environment: {e}")

        self.agent = AIPlan4GridAgent(
            env=self.env,
            scenario_id=self.scenario_id,
            tactical_horizon=self.tactical_horizon,
            solver=self.solver,
            test=self.test,
            debug=self.debug,
            _nb_gen_actions=self._nb_gen_actions,
            _nb_sto_actions=self._nb_sto_actions,
        )

    def launch(self, save: bool = False):
        """Launch the agent on the environment.

        Args:
            save (bool, optional): Whether to save the results or not. Defaults to False.
        """
        if self.debug:
            self.agent.print_summary()
            self.agent.print_grid_properties()
        nb_steps = self.strategic_horizon // self.tactical_horizon
        print(
            f"\nRunning the agent on scenario {self.scenario_id} for {nb_steps} steps of {self.tactical_horizon*self.time_step} minutes.\n"
        )
        cumulative_reward = 0
        beg_ = perf_counter()
        for i in tqdm(range(1, nb_steps + 1), desc="Steps"):
            reward, time_act, done = self.agent.progress()
            cumulative_reward += reward
            if done and i != nb_steps:
                self.agent.logger.info("The agent didn't survive.")
                break
        end_ = perf_counter()
        print(f"\nTotal reward: {cumulative_reward}")
        self.agent.display_grid()
        print(f"Logs are available in {self.agent.log_file}")
        if save:
            self.agent.episode.set_game_over(self.env.nb_time_step)
            self.agent.episode.set_meta(
                self.env,
                self.env.nb_time_step,
                cumulative_reward,
                env_seed=None,
                agent_seed=None,
            )
            self.agent.episode.set_episode_times(self.env, time_act, beg_, end_)
            self.agent.episode.to_disk()
            print(f"Results are available in {cfg.AGENT_DIR}")
        logging.shutdown()
