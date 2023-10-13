import time

import grid2op
from grid2op.Backend import PandaPowerBackend
from grid2op.Chronics import FromHandlers
from grid2op.Chronics.handlers import (
    CSVHandler,
    DoNothingHandler,
    NoisyForecastHandler,
    PerfectForecastHandler,
)

import plan4grid.config as cfg
from plan4grid.AIPlan4GridAgent import AIPlan4GridAgent
from plan4grid.utils import clean_logs


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
            raise KeyError(
                f"The parameter {e} is missing in the configuration file. Please make sure that the configuration file contains only the following self.parameters: {cfg.self.parameters_LIST}."
            )

    def __init__(
        self,
        env_name: str,
        scenario_id: int,
        tactical_horizon: int,
        strategic_horizon: int = 288,
        time_step: int = 5,
        solver: str = "enhsp",
        noise: bool = True,
        debug: bool = False,
    ):
        self.env_name = env_name
        self.scenario_id = scenario_id
        self.tactical_horizon = tactical_horizon
        self.strategic_horizon = strategic_horizon
        self.time_step = time_step
        self.solver = solver
        self.noise = noise
        self.debug = debug

        self.parameters = {
            cfg.TACTICAL_HORIZON: self.tactical_horizon,
            cfg.STRATEGIC_HORIZON: self.strategic_horizon,
            cfg.SOLVER: self.solver,
            cfg.NOISE: self.noise,
        }
        self._check_parameters()
        clean_logs()

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

    def launch(self):
        """Launch the agent."""
        env = grid2op.make(
            dataset=self.env_name,
            data_feeding_kwargs=self._get_data_feeding_kwargs(),
            test=self.debug,
            backend=PandaPowerBackend(),
        )
        agent = AIPlan4GridAgent(
            env=env,
            scenario_id=self.scenario_id,
            tactical_horizon=self.tactical_horizon,
            solver=self.solver,
            debug=self.debug,
        )
        agent.print_summary()
        agent.print_grid_properties()
        nb_steps = self.strategic_horizon // self.tactical_horizon
        print(f"Running the agent on scenario {self.scenario_id} for {nb_steps} steps...")
        cumulative_reward = 0
        for i in range(1, nb_steps + 1):
            print(f"\n* Episode {i}/{nb_steps}:")
            obs, reward, done, *_ = agent.progress(i)
            print(f"\tReward: {reward}")
            cumulative_reward += reward
            if done and i != (nb_steps):
                print("The episode is done before the end of the strategic horizon!")
                break
        time.sleep(2)
        agent.display_grid()
        print(f"\n* Cumulative reward: {cumulative_reward}")
