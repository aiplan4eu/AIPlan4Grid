AIPlan4Grid CLI
===============

Requirements
------------

.. note:: Python 3.10.0 is required.

.. code:: bash

   python -m pip install git+https://github.com/aiplan4eu/AIPlan4Grid.git@<version>

Where `<version>` is the version of the release you want to install.

Main usage
----------

.. code:: bash

   python -m plan4grid -h
   usage: python -m plan4grid [-h] -e <env_name> -s <scenario_id> [-c <config_file>] [-d] [--save]

   Run the AIPlan4Grid agent on the given environment with the given scenario. If no configuration file is given, the default configuration file will be used.

   optional arguments:
   -h, --help            show this help message and exit
   -e ENV_NAME, --env-name ENV_NAME
                           Name of the environment to use.
   -s SCENARIO_ID, --scenario-id SCENARIO_ID
                           ID of the scenario to use.
   -c CONFIG_FILE, --config-file CONFIG_FILE
                           Configuration file path. If not given, the default configuration file will be used.
   -d, --debug           Debug mode.
   --save                Save the results.

Configuration file
------------------

A configuration file can be provided to the agent. If no configuration file is provided, the default configuration file will be used. The default configuration file is `parameters.ini`.

It looks like this:

::

   [Parameters]
   tactical_horizon = 1 # The number of steps to look ahead in the future
   strategic_horizon = 288 # The number of time steps over which the agent is operated
   solver = enhsp # The solver to use (currently only enhsp is supported)
   noise = False # Whether to add noise to the observations or not
   test = False # Whether to run the agent in test mode or not