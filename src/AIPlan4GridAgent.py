import logging
from typing import Optional

import pandapower as pp
from grid2op.Action import BaseAction
from grid2op.Agent import BaseAgent
from grid2op.Environment import Environment
from grid2op.Observation import BaseObservation
from pandapower.pd2ppc import _pd2ppc
from pandapower.pypower.makePTDF import makePTDF

from UnifiedPlanningProblem import UnifiedPlanningProblem
from timeit import default_timer as timer


class AIPlan4GridAgent(BaseAgent):
    def _get_ptdf(self):
        net = self.grid
        pp.rundcpp(net)
        _, ppci = _pd2ppc(net)
        ptdf = makePTDF(ppci["baseMVA"], ppci["bus"], ppci["branch"])
        return ptdf

    def _get_grid_params(self):
        grid_params = {"gens": {}, "storages": {}, "lines": {}}

        # Generators parameters
        grid_params["gens"]["pmin"] = self.env.gen_pmin
        grid_params["gens"]["pmax"] = self.env.gen_pmax
        grid_params["gens"]["redispatchable"] = self.env.gen_redispatchable
        grid_params["gens"]["max_ramp_up"] = self.env.gen_max_ramp_up
        grid_params["gens"]["max_ramp_down"] = self.env.gen_max_ramp_down
        grid_params["gens"]["gen_cost_per_MW"] = self.env.gen_cost_per_MW
        grid_params["gens"]["slack"] = self.grid.gen["slack"].to_numpy()
        grid_params["gens"]["bus"] = self.grid.gen["bus"].to_numpy()

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

        # Lines parameters
        transmission_lines = self.grid.line[["from_bus", "to_bus"]]
        for tl_idx in transmission_lines.index:
            grid_params["lines"][tl_idx] = {
                "from": transmission_lines.at[tl_idx, "from_bus"],
                "to": transmission_lines.at[tl_idx, "to_bus"],
            }

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
        data_generator,
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
        self.data_generator = data_generator
        self.grid = self.env.backend._grid
        self.ptdf = self._get_ptdf()
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
            self.ptdf,
            self.grid_params,
            self.init_states,
        )

    def act(
        self, obs: BaseObservation, reward: float = 1.0, done: bool = False
    ) -> BaseAction:
        print("Creating unified planning problem")
        upb = self._to_unified_planning()
        print("Solving unified planning problem")
        start = timer()
        upb.solve()
        end = timer()
        print(f"Problem solved in {end - start} seconds")

    def step():
        pass
