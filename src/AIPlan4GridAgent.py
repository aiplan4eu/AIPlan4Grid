from math import atan, cos, sqrt
from timeit import default_timer as timer

import numpy as np
from grid2op.Environment import Environment
from pandapower.pd2ppc import _pd2ppc
from pandapower.pypower.makePTDF import makePTDF

import config as cfg
from UnifiedPlanningProblem import UnifiedPlanningProblem


class AIPlan4GridAgent:
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

        max_flows = np.array(
            (
                (
                    self.env.backend.lines_or_pu_to_kv
                    * self.env.backend.get_thermal_limit()
                    / 1000
                )
                * sqrt(3)
                * cos(atan(0.4))
            ),
            dtype=float,
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

    def get_ptdf(self):
        net = self.grid
        _, ppci = _pd2ppc(net)
        ptdf = makePTDF(ppci["baseMVA"], ppci[cfg.BUS], ppci["branch"])
        return ptdf

    def __init__(
        self,
        env: Environment,
        operational_horizon: int,
        solver: str,
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

        self.env = env
        self.env.set_id(2)
        self.operational_horizon = operational_horizon
        self.grid = self.env.backend._grid
        self.grid_params = self._get_grid_params()
        self.ptdf = self.get_ptdf()
        self.curr_obs = self.env.reset()
        self.solver = solver

    def display_grid(self):
        import matplotlib.pyplot as plt
        from grid2op.PlotGrid import PlotMatplot

        plot_helper = PlotMatplot(self.env.observation_space)
        obs = self.env.reset()
        plot_helper.plot_obs(obs)
        plt.show()

    def get_states(self):
        do_nothing_action = self.env.action_space({})

        simulated_observations = [self.curr_obs.simulate(do_nothing_action)[0]]
        for _ in range(self.operational_horizon - 1):
            obs, *_ = simulated_observations[-1].simulate(do_nothing_action)
            simulated_observations.append(obs)

        forecasted_states = {
            cfg.GENERATORS: np.array(
                [
                    simulated_observations[t].gen_p
                    for t in range(self.operational_horizon)
                ]
            ),
            cfg.LOADS: np.array(
                [
                    simulated_observations[t].load_p
                    for t in range(self.operational_horizon)
                ]
            ),
            cfg.STORAGES: np.array(
                [
                    simulated_observations[t].storage_charge
                    for t in range(self.operational_horizon)
                ]
            ),
            cfg.FLOWS: np.array(
                [
                    simulated_observations[t].p_or
                    for t in range(self.operational_horizon)
                ]
            ),
            cfg.TRANSMISSION_LINES: np.array(
                [
                    simulated_observations[t].rho >= 1
                    for t in range(self.operational_horizon)
                ]
            ),
        }

        initial_states = {
            cfg.GENERATORS: self.curr_obs.gen_p,
            cfg.LOADS: self.curr_obs.load_p,
            cfg.STORAGES: self.curr_obs.storage_charge,
            cfg.FLOWS: self.curr_obs.p_or,
            cfg.TRANSMISSION_LINES: self.curr_obs.rho >= 1,
        }

        return initial_states, forecasted_states

    def update_states(self):
        self.initial_states, self.forecasted_states = self.get_states()

    def up_actions_to_g2op_actions(self, up_actions):
        # first we create the dict that will be returned
        g2op_actions = {cfg.REDISPATCH: [], cfg.SET_STORAGE: []}

        # then we parse the up actions
        for action in up_actions:
            action = action.action.name
            # we split the string to extract the information
            action_info = action.split("_")
            # we get the type of the action
            action_type = action_info[0]
            # we get the id of the generator or the storage
            id = int(action_info[2])
            # we get the time step
            time_step = int(action_info[3])
            # we get the value of the action
            value = float(action_info[4])

            # we get the current value of the generator or the storage
            if action_type == cfg.GENERATOR_ACTION_PREFIX:
                current_value = self.curr_obs.gen_p[id]
            elif action_type == cfg.STORAGE_ACTION_PREFIX:
                current_value = self.curr_obs.storage_charge[id]
            else:
                raise RuntimeError(
                    "The action type is not valid, it should be either prod_target or storage_target"
                )

            # we compute the value of the action
            action_value = value - current_value

            # we add the action to the dict
            if action_type == cfg.GENERATOR_ACTION_PREFIX:
                g2op_actions[cfg.REDISPATCH].append((id, action_value))
            elif action_type == cfg.STORAGE_ACTION_PREFIX:
                g2op_actions[cfg.SET_STORAGE].append((id, action_value))
            else:
                raise RuntimeError(
                    "The action type is not valid, it should be either prod_target or storage_target"
                )

        return g2op_actions

    def get_actions(self, step: int):
        self.update_states()
        print("\tCreating UP problem...")
        upp = UnifiedPlanningProblem(
            self.operational_horizon,
            self.ptdf,
            self.grid_params,
            self.initial_states,
            self.forecasted_states,
            self.solver,
            id=step,
        )
        print(f"\tSaving UP problem in {cfg.LOG_DIR}")
        upp.save_problem()
        print("\tSolving UP problem...")
        start = timer()
        up_plan = upp.solve(simulate=True)
        end = timer()
        print(f"\tProblem solved in {end - start} seconds")
        if len(up_plan) == 0:
            return self.env.action_space({})
        g2op_actions = self.up_actions_to_g2op_actions(up_plan)
        return self.env.action_space(g2op_actions)

    def progress(self, step: int):
        actions = self.get_actions(step)
        observation, reward, done, info = self.env.step(actions)
        self.curr_obs = observation
        self.done = done
        return observation, reward, done, info
