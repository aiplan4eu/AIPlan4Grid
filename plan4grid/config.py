import os
from os.path import join as pjoin
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Parameters of the agent
DEFAULT_INI_FILE = pjoin(BASE_DIR, "parameters.ini")
PARAMETERS = "Parameters"
TACTICAL_HORIZON = "tactical_horizon"
STRATEGIC_HORIZON = "strategic_horizon"
SOLVER = "solver"
NOISE = "noise"
TEST = "test"

PARAMETERS_LIST = [TACTICAL_HORIZON, STRATEGIC_HORIZON, SOLVER, NOISE]
SOLVERS = ["enhsp"]

# DO NOT CHANGE THE FOLLOWING STRINGS (USED BY THE BACKEND)
###########################################################
SLACK = "slack"
BUS = "bus"
STORAGE_TO_SUBID = "storage_to_subid"
GEN_TO_SUBID = "gen_to_subid"

GENERATORS = "generators"
PMIN = "pmin"
PMAX = "pmax"
REDISPATCHABLE = "redispatchable"
MAX_RAMP_UP = "max_ramp_up"
MAX_RAMP_DOWN = "max_ramp_down"
GEN_COST_PER_MW = "gen_cost_per_MW"
GEN_BUS = "gen_bus"
GEN_PROD = "gen_prod"

STORAGES = "storages"
EMAX = "Emax"
EMIN = "Emin"
LOSS = "loss"
CHARGING_EFFICIENCY = "charging_efficiency"
DISCHARGING_EFFICIENCY = "discharging_efficiency"
STORAGE_MAX_P_PROD = "storage_max_p_prod"
STORAGE_MAX_P_ABSORB = "storage_max_p_absorb"
STORAGE_COST_PER_MW = "storage_marginal_cost"
STORAGE_BUS = "storage_bus"
STO_CHARGE = "sto_charge"

TRANSMISSION_LINES = "transmission_lines"
FLOWS = "flows"
FROM_BUS = "from_bus"
TO_BUS = "to_bus"
HV_BUS = "hv_bus"
LV_BUS = "lv_bus"
CONNECTED_STATUS = "connected_status"
CONGESTED_STATUS = "congested_status"
MAX_FLOW = "max_flow"

LOADS = "loads"

REDISPATCH = "redispatch"
SET_STORAGE = "set_storage"
###########################################################

RESULTS_DIR = pjoin(BASE_DIR, "results")
AGENT_DIR = pjoin(RESULTS_DIR, "AIPlan4GridAgent")
os.makedirs(AGENT_DIR, exist_ok=True)

LOG_DIR = pjoin(BASE_DIR, "log")
UPP_SUFFIX = ".upp"
PDDL_SUFFIX = ".pddl"
LOG_SUFFIX = ".log"

GENERATOR_ACTION_PREFIX = "gen"
STORAGE_ACTION_PREFIX = "sto"
ADVANCE_STEP_ACTION = "advance_step"
INCREASE_ACTION = "increase"
DECREASE_ACTION = "decrease"
DIRECTIONS = [INCREASE_ACTION, DECREASE_ACTION]
