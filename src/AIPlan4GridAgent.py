from grid2op.Agent import BaseAgent
from grid2op.Environment import Environment
from grid2op.Action import BaseAction
from grid2op.Observation import BaseObservation
from typing import Optional
import logging
from pandapower.pd2ppc import _pd2ppc
from pandapower.pypower.makePTDF import makePTDF
from UnifiedPlanningProblem import UnifiedPlanningProblem


# NOTE: for now there is a lot of 'magic strings' in the code due to the massive utilisation of Pandas DataFrames by the backend (PandaPower) of grid2op
# so maybe we should consider to create a config.py file to store all the constants?


class AIPlan4GridAgent(BaseAgent):
    def _get_ptdf(self):
        net = self.grid
        # pp.rundcpp(net)
        _, ppci = _pd2ppc(net)
        ptdf = makePTDF(ppci["baseMVA"], ppci["bus"], ppci["branch"])
        return ptdf

    def _get_mapping(self):
        # I could use the following grid2op function https://grid2op.readthedocs.io/en/latest/observation.html#grid2op.Observation.BaseObservation.flow_bus_matrix
        # but I don't want to be dependant of a function that gives me too much information
        mapping = {"buses": {}, "transmission_lines": {}}
        buses = self.grid.bus.loc[self.grid.bus["in_service"]]

        for bus_idx in buses.index:
            mapping["buses"][bus_idx] = {
                "gen_idx": [],
                "storage_idx": [],
                "load_idx": [],
            }

        gens = self.grid.gen["bus"]
        storages = self.grid.storage["bus"]
        loads = self.grid.load["bus"]

        for i, gen_bus in enumerate(gens):
            mapping["buses"].setdefault(gen_bus, {"gen_idx": []})["gen_idx"].append(i)

        for i, storage_bus in enumerate(storages):
            mapping["buses"].setdefault(storage_bus, {"storage_idx": []})[
                "storage_idx"
            ].append(i)

        for i, load_bus in enumerate(loads):
            mapping["buses"].setdefault(load_bus, {"load_idx": []})["load_idx"].append(
                i
            )

        transmission_lines = self.grid.line[["from_bus", "to_bus"]]

        for tl in transmission_lines.index:
            mapping["transmission_lines"][tl] = {
                "from": transmission_lines.at[tl, "from_bus"],
                "to": transmission_lines.at[tl, "to_bus"],
            }

        return mapping

    def _get_grid_params(self):
        grid_params = {"gens": {}, "storages": {}}

        # Generators parameters
        grid_params["gens"]["pmin"] = self.env.gen_pmin
        grid_params["gens"]["pmax"] = self.env.gen_pmax
        grid_params["gens"]["redispatchable"] = self.env.gen_redispatchable
        grid_params["gens"]["max_ramp_up"] = self.env.gen_max_ramp_up
        grid_params["gens"]["max_ramp_down"] = self.env.gen_max_ramp_down
        grid_params["gens"]["gen_cost_per_MW"] = self.env.gen_cost_per_MW

        # Storages parameters
        grid_params["storages"]["Emax"] = self.env.storage_Emax
        grid_params["storages"]["Emin"] = self.env.storage_Emin
        grid_params["storages"]["loss"] = self.env.storage_loss
        grid_params["storages"][
            "charging_efficiency"
        ] = self.env.storage_charging_efficiency
        grid_params["storages"][
            "discharging_efficiency"
        ] = self.env.storage_discharging_efficiency

        return grid_params

    def _get_initial_states(self):
        init_states = {}
        init_states["gens"] = self.env.current_obs.gen_p
        init_states["loads"] = self.env.current_obs.load_p
        init_states["storages"] = self.env.current_obs.storage_charge
        flow_mat, _ = self.env.current_obs.flow_bus_matrix()
        init_states["flows"] = flow_mat
        return init_states

    def __init__(
        self,
        env: Environment,
        horizon: int,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        if env.n_storage > 0 and not env.action_space.supports_type("set_storage"):
            raise RuntimeError(
                "Impossible to create this class with an environment that does not allow "
                "modification of storage units when there are storage units on the grid. "
            )
        if not env.action_space.supports_type("redispatch"):
            raise RuntimeError(
                "This type of agent can only perform actions using storage units, curtailment or"
                "redispatching. It requires at least to be able to do redispatching."
            )
        super().__init__(env.action_space)
        self.env = env
        self.horizon = horizon
        self.grid = self.env.backend._grid
        self.nb_gens = self.grid.gen.shape[0]
        self.nb_storages = self.grid.storage.shape[0]
        self.ptdf = self._get_ptdf()
        self.mapping = self._get_mapping()
        self.grid_params = self._get_grid_params()
        self.init_states = self._get_initial_states()

        if logger is None:
            self.logger: logging.Logger = logging.getLogger(__name__)
            self.logger.disabled = True
            # self.logger.disabled = False
            # self.logger.addHandler(logging.StreamHandler(sys.stdout))
            # self.logger.setLevel(level=logging.DEBUG)
        else:
            self.logger: logging.Logger = logger.getChild("AIPlan4GridAgent")

    def _to_unified_planning(self):
        return UnifiedPlanningProblem(
            self.horizon,
            self.nb_gens,
            self.nb_storages,
            self.ptdf,
            self.mapping,
            self.grid_params,
            self.init_states,
        )

    def act(
        self, obs: BaseObservation, reward: float = 1.0, done: bool = False
    ) -> BaseAction:
        upb = self._to_unified_planning().create_problem()

    def step():
        pass
