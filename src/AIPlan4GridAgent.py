from math import atan, cos, sqrt
from timeit import default_timer as timer

import numpy as np
from grid2op.Agent import BaseAgent
from grid2op.Environment import Environment
from pandapower.pd2ppc import _pd2ppc
from pandapower.pypower.makePTDF import makePTDF

import config as cfg
from UnifiedPlanningProblem import UnifiedPlanningProblem
from utils import verbose_print


class AIPlan4GridAgent(BaseAgent):
    def _get_ptdf(self):
        net = self.grid
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
        power_lines = self.grid.line[[cfg.FROM_BUS, cfg.TO_BUS]]
        transfo_lines = self.grid.trafo[[cfg.HV_BUS, cfg.LV_BUS]]

        max_flows = (
            (
                self.env.backend.lines_or_pu_to_kv
                * self.env.backend.get_thermal_limit()
                / 1000
            )
            * sqrt(3)
            * cos(atan(0.4))
        )  # from Ampere to MW

        for tl_idx in power_lines.index:
            grid_params[cfg.TRANSMISSION_LINES][tl_idx] = {
                cfg.FROM_BUS: power_lines.at[tl_idx, cfg.FROM_BUS],
                cfg.TO_BUS: power_lines.at[tl_idx, cfg.TO_BUS],
            }
        for tl_idx in transfo_lines.index:
            grid_params[cfg.TRANSMISSION_LINES][len(power_lines.index) + tl_idx] = {
                cfg.FROM_BUS: transfo_lines.at[tl_idx, cfg.HV_BUS],
                cfg.TO_BUS: transfo_lines.at[tl_idx, cfg.LV_BUS],
            }

        for tl_index in grid_params[cfg.TRANSMISSION_LINES].keys():
            grid_params[cfg.TRANSMISSION_LINES][tl_index][
                cfg.STATUS
            ] = self.env.backend.get_line_status()[tl_index]
            grid_params[cfg.TRANSMISSION_LINES][tl_index][cfg.MAX_FLOW] = max_flows[
                tl_index
            ]

        return grid_params

    def _get_states(self):
        base_obs = self.env.reset()
        dn_act = self.env.action_space({})

        sim_obs = [base_obs.simulate(dn_act)[0]]
        for _ in range(self.horizon - 1):
            obs, *_ = sim_obs[-1].simulate(dn_act)
            sim_obs.append(obs)

        forecasted_states = {
            cfg.GENERATORS: np.array([sim_obs[t].gen_p for t in range(self.horizon)]),
            cfg.LOADS: np.array([sim_obs[t].load_p for t in range(self.horizon)]),
            cfg.STORAGES: np.array(
                [sim_obs[t].storage_charge for t in range(self.horizon)]
            ),
            cfg.FLOWS: np.array([sim_obs[t].p_or for t in range(self.horizon)]),
            cfg.TRANSMISSION_LINES: np.array(
                [sim_obs[t].rho >= 1 for t in range(self.horizon)]
            ),
        }

        initial_states = {
            cfg.GENERATORS: base_obs.gen_p,
            cfg.LOADS: base_obs.load_p,
            cfg.STORAGES: base_obs.storage_charge,
            cfg.FLOWS: base_obs.p_or,
            cfg.TRANSMISSION_LINES: base_obs.rho >= 1,
        }

        return initial_states, forecasted_states

    def __init__(
        self,
        env: Environment,
        horizon: int,
        solver: str,
        verbose: bool,
    ):
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
        self.env.set_id(0)
        self.horizon = horizon
        self.grid = self.env.backend._grid
        self.grid_params = self._get_grid_params()
        self.initial_states, self.forecasted_states = self._get_states()
        self.solver = solver

        self._VERBOSE = verbose
        global vprint
        vprint = verbose_print(self._VERBOSE)

    def display_grid(self):
        import matplotlib.pyplot as plt
        from grid2op.PlotGrid import PlotMatplot

        plot_helper = PlotMatplot(self.env.observation_space)
        obs = self.env.reset()
        plot_helper.plot_obs(obs)
        plt.show()

    def act(self):
        self.ptdf = self._get_ptdf()
        vprint("Creating UP problem...")
        upp = UnifiedPlanningProblem(
            self.horizon,
            self.ptdf,
            self.grid_params,
            self.initial_states,
            self.forecasted_states,
            self.solver,
        )
        vprint(f"Saving UP problem in {cfg.TMP_DIR}")
        upp.save_problem()
        vprint("Solving UP problem...")
        start = timer()
        upp.solve()
        end = timer()
        vprint(f"Problem solved in {end - start} seconds")

    def step():
        pass
