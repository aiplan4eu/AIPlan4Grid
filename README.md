# AIPlan4Grid

This repository contains the code of the AIPlan4Grid agent, which is an agent that uses the [UnifiedPlanning](https://github.com/aiplan4eu/unified-planning) optimization framework in order to solve the [Grid2Op](https://github.com/rte-france/Grid2Op) power grid control problem.

## Installation of the package

Python 3.10.0 is required.

You can install the package with:

```bash
python -m pip install git+https://github.com/aiplan4eu/AIPlan4Grid.git@<version>
```

Where `<version>` is the version of the released package you want to install.

Or you can also install the package from the wheel file located in the `dist` folder after building the package. See the [Building the package](#building-the-package) section for more information.

## Main usage

### CLI

```bash
python -m plan4grid -h
usage: python -m plan4grid [-h] -e <env_name> -s <scenario_id> [-c <config_file>] [-d]

Run the AIPlan4Grid agent on the given environment with the given scenario. If no configuration file is given, the default configuration file will be used.

options:
  -h, --help            show this help message and exit
  -e ENV_NAME, --env-name ENV_NAME
                        Name of the environment to use.
  -s SCENARIO_ID, --scenario-id SCENARIO_ID
                        ID of the scenario to use.
  -c CONFIG_FILE, --config-file CONFIG_FILE
                        Configuration file path. If not given, the default configuration file will be used.
  -d, --debug           Debug mode.
```

#### Configuration file

A configuration file can be provided to the agent. If no configuration file is provided, the default configuration file will be used. The default configuration file is `parameters.ini`.

It looks like this:

```ini
[Parameters]
tactical_horizon = 1 # The number of steps to look ahead in the future
strategic_horizon = 288 # The number of time steps over which the agent is operated
solver = enhsp # The solver to use (currently only enhsp is supported)
noise = False # Whether to add noise to the observations or not
test = False # Whether to run the agent in test mode or not
```

### Notebook

You can also run the agent from a notebook using the `Launcher` class. See the [notebooks](notebooks) folder and the API documentation for more information.

## Development requirements

In order to develop the package, you need to install the required dependencies for development. Moreover, it is highly recommended to use a virtual environment.

```bash
pip install -r requirements/requirements.txt
```

## Building documentation

Documentation can be built using sphinx and by installing required dependencies.

```bash
python -m pip install -r requirements/requirements_docs.txt
python -m sphinx docs/source docs/build/
```

## Building the package

Build, test and install the package with the following command:

```bash
python -m pip install -r requirements/requirements_build.txt
cd scripts
chmod +x build_pkg.sh
./build_pkg.sh
```

Note: if you want to update the version number of the package, you'll have to do it in the `setup.py` file **AND** in the `__init__.py` file of the `plan4grid` folder (the `__version__` variable) before building the package.

## Grid2Viz

After running the agent, you can visualize the actions he took and the grid states using the [Grid2Viz tool](https://github.com/rte-france/grid2viz).

```bash
cd scripts
chmod +x visualize.sh
./visualize.sh <g2op_env_path>
```

Where `<g2op_env_path>` is the path to the environment data directory stored by [Grid2Op](https://github.com/rte-france/Grid2Op) (stored in a folder called "`data_grid2op`").
