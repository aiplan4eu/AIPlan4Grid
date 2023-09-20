# AIPlan4Grid

## Requirements

Python 3.9.0 or higher.

```bash
pip install -r requirements/requirements.txt
```

## Main usage

```bash
python -m plan4grid -h
usage: [-h] [-c CONFIG_FILE]

optional arguments:
  -h, --help            show this help message and exit
  -c CONFIG_FILE, --config-file CONFIG_FILE
                        Configuration file path. If not given, the default configuration file will be used.
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
chmod +x build_pkg.sh
./build_pkg.sh
```
