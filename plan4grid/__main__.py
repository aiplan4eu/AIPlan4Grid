from argparse import ArgumentParser, Namespace
from configparser import ConfigParser
from os.path import join as pjoin

import plan4grid.config as cfg
from plan4grid.Launcher import Launcher
from plan4grid.utils import strtobool


def _fill_parser(parser: ArgumentParser):
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
    parser.add_argument(
        "-d",
        "--debug",
        help="Debug mode.",
        required=False,
        action="store_true",
    )
    parser.usage = "python -m plan4grid [-h] -e <env_name> -s <scenario_id> [-c <config_file>] [-d]"
    parser.description = "Run the AIPlan4Grid agent on the given environment with the given scenario. If no configuration file is given, the default configuration file will be used."


def _parse_ini(ini_file_path: str) -> dict:
    """Parse the given configuration file."""
    try:
        config = ConfigParser()
        config.read(ini_file_path)
        parameters_section = config[cfg.PARAMETERS]
        parameters = {
            cfg.TACTICAL_HORIZON: int(parameters_section[cfg.TACTICAL_HORIZON]),
            cfg.STRATEGIC_HORIZON: int(parameters_section[cfg.STRATEGIC_HORIZON]),
            cfg.SOLVER: parameters_section[cfg.SOLVER],
            cfg.NOISE: strtobool(parameters_section[cfg.NOISE]),
            cfg.TEST: strtobool(parameters_section[cfg.TEST]),
        }
        return parameters
    except Exception:
        raise Exception(f"An error occurred while parsing the configuration file: {ini_file_path}")


def main(args: Namespace):
    """Main function."""
    try:
        if args.config_file is None:
            ini_file_path = pjoin(cfg.BASE_DIR, cfg.DEFAULT_INI_FILE)
            parameters = _parse_ini(ini_file_path)
        else:
            parameters = _parse_ini(args.config_file)

        if str(args.scenario_id).isdigit():
            args.scenario_id = int(args.scenario_id)

        launcher = Launcher(
            env_name=args.env_name,
            scenario_id=args.scenario_id,
            tactical_horizon=parameters[cfg.TACTICAL_HORIZON],
            strategic_horizon=parameters[cfg.STRATEGIC_HORIZON],
            solver=parameters[cfg.SOLVER],
            noise=parameters[cfg.NOISE],
            test=parameters[cfg.TEST],
            debug=args.debug,
        )
        launcher.launch()
    except Exception as e:
        print(e)


if __name__ == "__main__":
    parser = ArgumentParser()
    _fill_parser(parser)
    args = parser.parse_args()
    main(args)
