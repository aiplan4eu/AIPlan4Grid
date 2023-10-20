# AIPlan4Grid

## Requirements

Python 3.10.0 is required.

```bash
pip install -r requirements/requirements.txt
```

## Main usage

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

### Building documentation

Documentation can be built using sphinx and by installing required dependencies.

```bash
python -m pip install -r requirements/requirements_docs.txt
python -m sphinx docs/source docs/build/
```

### Building the project

Build and install the project with the following command:

```bash
python -m pip install -r requirements/requirements_build.txt
chmod +x build_pkg.sh
./build_pkg.sh
```
