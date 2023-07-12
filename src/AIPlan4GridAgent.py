from timeit import default_timer as timer

import numpy as np
import pandapower as pp
from grid2op.Agent import BaseAgent
from grid2op.Environment import Environment
from pandapower.pd2ppc import _pd2ppc
from pandapower.pypower.makePTDF import makePTDF

import src.config as cfg
from src.utils import verbose_print
from UnifiedPlanningProblem import UnifiedPlanningProblem


class AIPlan4GridAgent(BaseAgent):
    def _get_ptdf(self):
        net = self.grid
        pp.rundcpp(net)
        _, ppci = _pd2ppc(net)
        ptdf = makePTDF(ppci["baseMVA"], ppci[cfg.BUS], ppci["branch"])
        return ptdf

    def _get_grid_params(self):
        grid_params = {cfg.GENERATORS: {}, cfg.STORAGES: {}, cfg.TRANSMISSION_LINES: {}}

        # Generators parameters
        grid_params[cfg.GENERATORS][cfg.PMIN] = self.env.gen_pmin
        grid_params[cfg.GENERATORS][cfg.PMAX] = self.env.gen_pmax
        grid_params[cfg.GENERATORS][cfg.REDISPATCHABLE] = self.env.gen_redispatchable
        grid_params[cfg.GENERATORS][cfg.MAX_RAMP_UP] = self.env.gen_max_ramp_up
        grid_params[cfg.GENERATORS][cfg.MAX_RAMP_DOWN] = self.env.gen_max_ramp_down
        grid_params[cfg.GENERATORS][cfg.GEN_COST_PER_MW] = self.env.gen_cost_per_MW
        grid_params[cfg.GENERATORS][cfg.SLACK] = self.grid.gen[cfg.SLACK].to_numpy()
        grid_params[cfg.GENERATORS][cfg.BUS] = self.grid.gen[cfg.BUS].to_numpy()

        # Storages parameters
        grid_params[cfg.STORAGES][cfg.EMAX] = self.env.storage_Emax
        grid_params[cfg.STORAGES][cfg.EMIN] = self.env.storage_Emin
        grid_params[cfg.STORAGES][cfg.LOSS] = self.env.storage_loss
        grid_params[cfg.STORAGES][
            cfg.CHARGING_EFFICIENCY
        ] = self.env.storage_charging_efficiency
        grid_params[cfg.STORAGES][
            cfg.DISCHARGING_EFFICIENCY
        ] = self.env.storage_discharging_efficiency

        # Lines parameters
        transmission_lines = self.grid.line[[cfg.FROM_BUS, cfg.TO_BUS]]
        for tl_idx in transmission_lines.index:
            grid_params[cfg.TRANSMISSION_LINES][tl_idx] = {
                cfg.FROM_BUS: transmission_lines.at[tl_idx, cfg.FROM_BUS],
                cfg.TO_BUS: transmission_lines.at[tl_idx, cfg.TO_BUS],
            }

        return grid_params

    def _get_reference_states(self):
        # TODO: to refactor with forcasted data
        reference_states = {
            cfg.GENERATORS: np.array(
                [self.env.current_obs.gen_p for _ in range(self.horizon)]
            ),
            cfg.LOADS: np.array(
                [self.env.current_obs.load_p for _ in range(self.horizon)]
            ),
            cfg.STORAGES: np.array(
                [self.env.current_obs.storage_charge for _ in range(self.horizon)]
            ),
            cfg.FLOWS: np.array(
                [self.env.current_obs.flow_bus_matrix()[0] for _ in range(self.horizon)]
            ),
        }
        return reference_states

    def __init__(
        self, env: Environment, horizon: int, solver: str, verbose: bool
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
        self.solver = solver

        self._VERBOSE = verbose
        global vprint
        vprint = verbose_print(self._VERBOSE)

    def act(self):
        vprint("Creating UP problem...")
        upb = UnifiedPlanningProblem(
            self.horizon,
            self.ptdf,
            self.grid_params,
            self.reference_states,
            self.solver,
        )
        vprint("Saving UP problem in tmp/problem.up")
        upb.save_problem()
        vprint("Solving UP problem...")
        start = timer()
        upb.solve()
        end = timer()
        vprint(f"Problem solved in {end - start} seconds")

    def step():
        pass
