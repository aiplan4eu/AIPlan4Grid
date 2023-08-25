import argparse
import configparser
from os.path import join as pjoin

import grid2op
from AIPlan4GridAgent import AIPlan4GridAgent
from grid2op.Backend import PandaPowerBackend

import config as cfg
import shutil


def clean_logs():
    """Remove recursively the log directory if it exists"""
    try:
        shutil.rmtree(cfg.LOG_DIR)
    except FileNotFoundError:
        pass


def fill_parser(parser: argparse.ArgumentParser):
    """Fill the given parser"""
    parser.add_argument(
        "-c",
        "--config-file",
        help="Configuration file path. If not given, the default configuration file will be used.",
        default=None,
        required=False,
    )


def parse_ini(ini_file_path: str) -> dict:
    """Parse the given configuration file"""
    config = configparser.ConfigParser()
    config.read(ini_file_path)

    parameters_section = config[cfg.PARAMETERS]
    parameters = {
        cfg.ENV_NAME: parameters_section[cfg.ENV_NAME],
        cfg.OPERATIONAL_HORIZON: int(parameters_section[cfg.OPERATIONAL_HORIZON]),
        cfg.TACTICAL_HORIZON: int(parameters_section[cfg.TACTICAL_HORIZON]),
        cfg.STRATEGIC_HORIZON: int(parameters_section[cfg.STRATEGIC_HORIZON]),
        cfg.DISCRETIZATION: int(parameters_section[cfg.DISCRETIZATION]),
        cfg.SOLVER: parameters_section[cfg.SOLVER],
        cfg.SCENARIO: int(parameters_section[cfg.SCENARIO]),
    }
    return parameters


def routine(agent: AIPlan4GridAgent):
    """Routine for the agent"""
    for i in range(STRATEGIC_HORIZON):
        print(f"\n* Episode {i}:")
        obs, reward, *_ = agent.progress(i)
        print(f"\tReward: {reward}")


def main(args: argparse.Namespace):
    try:
        if args.config_file is None:
            ini_file_path = pjoin(cfg.BASE_DIR, cfg.DEFAULT_INI_FILE)
            parameters = parse_ini(ini_file_path)
        else:
            parameters = parse_ini(args.config_file)
        global STRATEGIC_HORIZON
        STRATEGIC_HORIZON = parameters[cfg.STRATEGIC_HORIZON]
        env = grid2op.make(
            dataset=parameters[cfg.ENV_NAME],
            test=True,
            backend=PandaPowerBackend(),
        )
        agent = AIPlan4GridAgent(
            env=env,
            scenario_id=parameters[cfg.SCENARIO],
            operational_horizon=parameters[cfg.OPERATIONAL_HORIZON],
            discretization=parameters[cfg.DISCRETIZATION],
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
