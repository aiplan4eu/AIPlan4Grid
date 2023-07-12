import logging
from timeit import default_timer as timer
from typing import Optional

import numpy as np
import pandapower as pp
from grid2op.Action import BaseAction
from grid2op.Agent import BaseAgent
from grid2op.Environment import Environment
from grid2op.Observation import BaseObservation
from pandapower.pd2ppc import _pd2ppc
from pandapower.pypower.makePTDF import makePTDF

from UnifiedPlanningProblem import UnifiedPlanningProblem


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

    def _get_reference_states(self):
        # TODO: to refactor with forcasted data
        reference_states = {
            "gens": np.array([self.env.current_obs.gen_p for _ in range(self.horizon)]),
            "loads": np.array(
                [self.env.current_obs.load_p for _ in range(self.horizon)]
            ),
            "storages": np.array(
                [self.env.current_obs.storage_charge for _ in range(self.horizon)]
            ),
            "flows": np.array(
                [self.env.current_obs.flow_bus_matrix()[0] for _ in range(self.horizon)]
            ),
        }
        return reference_states

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
        self.ptdf = self._get_ptdf()
        self.grid_params = self._get_grid_params()
        self.reference_states = self._get_reference_states()

        if logger is None:
            self.logger: logging.Logger = logging.getLogger(__name__)
            self.logger.disabled = True
            # self.logger.disabled = False
            # self.logger.addHandler(logging.StreamHandler(sys.stdout))
            # self.logger.setLevel(level=logging.DEBUG)
        else:
            self.logger: logging.Logger = logger.getChild("AIPlan4GridAgent")

    def act(
        self, obs: BaseObservation, reward: float = 1.0, done: bool = False
    ) -> BaseAction:
        print("Creating unified planning problem")
        upb = UnifiedPlanningProblem(
            self.horizon,
            self.ptdf,
            self.grid_params,
            self.reference_states,
        )
        upb.print_summary()
        upb.print_problem()
        print("Solving unified planning problem")
        start = timer()
        upb.solve()
        end = timer()
        print(f"Problem solved in {end - start} seconds")

    def step():
        pass
