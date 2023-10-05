import argparse
import configparser
import shutil
import time
from os.path import join as pjoin

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
from plan4grid.utils import strtobool


def clean_logs():
    """Remove recursively the log directory if it exists."""
    try:
        shutil.rmtree(cfg.LOG_DIR)
    except FileNotFoundError:
        pass


def fill_parser(parser: argparse.ArgumentParser):
    """Fill the given parser."""
    parser.add_argument(
        "-e",
        "--env-name",
        help="Name of the environment to use.",
        required=True,
    )
    parser.add_argument(
        "-s",
        "--scenario-id",
        help="ID of the scenario to use.",
        required=True,
    )
    parser.add_argument(
        "-c",
        "--config-file",
        help="Configuration file path. If not given, the default configuration file will be used.",
        default=None,
        required=False,
    )
    parser.usage = (
        "python -m plan4grid [-h] -e <env_name> -s <scenario_id> [-c <config_file>]"
    )
    parser.description = "Run the AIPlan4Grid agent on the given environment with the given scenario. If no configuration file is given, the default configuration file will be used."


def parse_ini(ini_file_path: str) -> dict:
    """Parse the given configuration file."""
    config = configparser.ConfigParser()
    config.read(ini_file_path)

    parameters_section = config[cfg.PARAMETERS]
    parameters = {
        cfg.TACTICAL_HORIZON: int(parameters_section[cfg.TACTICAL_HORIZON]),
        cfg.STRATEGIC_HORIZON: int(parameters_section[cfg.STRATEGIC_HORIZON]),
        cfg.SOLVER: parameters_section[cfg.SOLVER],
        cfg.NOISE: strtobool(parameters_section[cfg.NOISE]),
    }
    return parameters


def routine(agent: AIPlan4GridAgent):
    """Routine for the agent."""
    agent.print_summary()
    agent.print_grid_properties()
    nb_steps = STRATEGIC_HORIZON // TACTICAL_HORIZON
    print(f"Running the agent on scenario {agent.scenario_id} for {nb_steps} steps...")
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


def get_data_feeding_kwargs(time_step: int, tactical_horizon: int, noisy: bool) -> dict:
    if noisy:
        handler = NoisyForecastHandler
    else:
        handler = PerfectForecastHandler

    return {
        "gridvalueClass": FromHandlers,
        "gen_p_handler": CSVHandler("prod_p"),
        "load_p_handler": CSVHandler("load_p"),
        "gen_v_handler": DoNothingHandler("prod_v"),
        "load_q_handler": CSVHandler("load_q"),
        "h_forecast": [h * time_step for h in range(tactical_horizon)],
        "gen_p_for_handler": handler("prod_p_forecasted"),
        "load_p_for_handler": handler("load_p_forecasted"),
        "load_q_for_handler": handler("load_q_forecasted"),
    }


def check_parameters(parameters: dict):
    """Check the given parameters."""
    try:
        if parameters[cfg.TACTICAL_HORIZON] > parameters[cfg.STRATEGIC_HORIZON]:
            raise ValueError(
                f"The tactical horizon ({parameters[cfg.TACTICAL_HORIZON]}) cannot be greater than the strategic horizon ({parameters[cfg.STRATEGIC_HORIZON]})."
            )
        if parameters[cfg.TACTICAL_HORIZON] <= 0:
            raise ValueError(
                f"The tactical horizon ({parameters[cfg.TACTICAL_HORIZON]}) must be greater than 0."
            )
        if parameters[cfg.STRATEGIC_HORIZON] <= 0:
            raise ValueError(
                f"The strategic horizon ({parameters[cfg.STRATEGIC_HORIZON]}) must be greater than 0."
            )
        if parameters[cfg.SOLVER] not in cfg.SOLVERS:
            raise ValueError(
                f"The solver ({parameters[cfg.SOLVER]}) must be one of the following: {cfg.SOLVERS}."
            )
        if parameters[cfg.NOISE] not in [True, False]:
            raise ValueError(
                f"The noise parameter ({parameters[cfg.NOISE]}) must be a boolean."
            )
        if parameters[cfg.STRATEGIC_HORIZON] > 288:
            raise ValueError(
                f"The strategic horizon ({parameters[cfg.STRATEGIC_HORIZON]}) must be lower or equal to 288, (24 hours in 5 minutes time steps)."
            )
    except KeyError as e:
        raise KeyError(
            f"The parameter {e} is missing in the configuration file. Please make sure that the configuration file contains only the following parameters: {cfg.PARAMETERS_LIST}."
        )


def main(args: argparse.Namespace):
    """Main function."""
    try:
        if args.config_file is None:
            ini_file_path = pjoin(cfg.BASE_DIR, cfg.DEFAULT_INI_FILE)
            parameters = parse_ini(ini_file_path)
        else:
            parameters = parse_ini(args.config_file)

        check_parameters(parameters)

        global STRATEGIC_HORIZON
        global TACTICAL_HORIZON

        STRATEGIC_HORIZON = parameters[cfg.STRATEGIC_HORIZON]
        TACTICAL_HORIZON = parameters[cfg.TACTICAL_HORIZON]
        TIME_STEP = 5

        env = grid2op.make(
            dataset=args.env_name,
            data_feeding_kwargs=get_data_feeding_kwargs(
                TIME_STEP,
                TACTICAL_HORIZON,
                parameters[cfg.NOISE],
            ),
            test=True,
            backend=PandaPowerBackend(),
        )
        agent = AIPlan4GridAgent(
            env=env,
            scenario_id=int(args.scenario_id),
            tactical_horizon=TACTICAL_HORIZON,
            solver=parameters[cfg.SOLVER],
            debug=True,
        )
        clean_logs()
        routine(agent)
    except Exception as e:
        print(e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    fill_parser(parser)
    args = parser.parse_args()
    main(args)
