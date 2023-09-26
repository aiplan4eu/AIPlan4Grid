from pathlib import Path
from os.path import join as pjoin

BASE_DIR = Path(__file__).resolve().parent.parent

DEFAULT_INI_FILE = "parameters.ini"
PARAMETERS = "Parameters"
OPERATIONAL_HORIZON = "operational_horizon"
TACTICAL_HORIZON = "tactical_horizon"
STRATEGIC_HORIZON = "strategic_horizon"
SOLVER = "solver"

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

TRANSMISSION_LINES = "transmission_lines"
FLOWS = "flows"
FROM_BUS = "from_bus"
TO_BUS = "to_bus"
HV_BUS = "hv_bus"
LV_BUS = "lv_bus"
STATUS = "status"
MAX_FLOW = "max_flow"

LOADS = "loads"

REDISPATCH = "redispatch"
SET_STORAGE = "set_storage"
###########################################################

LOG_DIR = pjoin(BASE_DIR, "log")
UPP_SUFFIX = ".upp"
PDDL_SUFFIX = ".pddl"
LOG_SUFFIX = ".log"
OUT_FILE = "out.log"
WARN_FILE = "warning.log"

GENERATOR_ACTION_PREFIX = "gen"
STORAGE_ACTION_PREFIX = "sto"
