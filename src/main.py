import argparse
import configparser
from os.path import join as pjoin

import grid2op
from AIPlan4GridAgent import AIPlan4GridAgent
from grid2op.Backend import PandaPowerBackend
from grid2op.Environment import Environment

import config as cfg


def fill_parser(parser: argparse.ArgumentParser):
    """Fill the given parser"""
    parser.add_argument(
        "-c",
        "--config-file",
        help="Configuration file path",
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
        cfg.HORIZON: int(parameters_section[cfg.HORIZON]),
        cfg.SOLVER: parameters_section[cfg.SOLVER],
    }
    return parameters


def _routine(env: Environment, agent: AIPlan4GridAgent):
    """Routine for the agent"""
    agent.act()


def main(args: argparse.Namespace):
    try:
        if args.config_file is None:
            ini_file_path = pjoin(cfg.BASE_DIR, cfg.DEFAULT_INI_FILE)
            parameters = parse_ini(ini_file_path)
        else:
            parameters = parse_ini(args.config_file)
        env = grid2op.make(
            parameters[cfg.ENV_NAME], test=True, backend=PandaPowerBackend()
        )
        agent = AIPlan4GridAgent(
            env,
            parameters[cfg.HORIZON],
            solver=parameters[cfg.SOLVER],
            verbose=True,
        )
        _routine(env, agent)
    except Exception as e:
        print(e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    fill_parser(parser)
    args = parser.parse_args()
    main(args)
