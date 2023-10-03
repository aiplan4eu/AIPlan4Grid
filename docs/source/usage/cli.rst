AIPlan4Grid CLI
===============

Requirements
------------

Python 3.10.0 or higher.

.. code:: bash

   pip install plan4grid

Main usage
----------

.. code:: bash

   python -m plan4grid -h
   usage: python -m plan4grid [-h] -e <env_name> -s <scenario_id> [-c <config_file>]

   Run the AIPlan4Grid agent on the given environment with the given scenario. If no configuration file is given, the default configuration file will be used.

   options:
   -h, --help            show this help message and exit
   -e ENV_NAME, --env-name ENV_NAME
                           Name of the environment to use.
   -s SCENARIO_ID, --scenario-id SCENARIO_ID
                           ID of the scenario to use.
   -c CONFIG_FILE, --config-file CONFIG_FILE
                           Configuration file path. If not given, the default configuration file will be used.

