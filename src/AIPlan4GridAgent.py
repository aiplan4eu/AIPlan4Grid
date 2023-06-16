from grid2op.Agent import BaseAgent
from grid2op.Environment import Environment
from grid2op.Action import ActionSpace, BaseAction
from grid2op.Observation import BaseObservation
from typing import Optional
import logging
import numpy as np


class AIPlan4GridAgent(BaseAgent):
    def __init__(
        self,
        observation_space: BaseObservation,
        action_space: ActionSpace,
        env: Environment,
        margin_sparse: float = 5e-3,
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
        BaseAgent.__init__(self, action_space)
        self.obs_space = observation_space
        self.margin_sparse = margin_sparse

        if logger is None:
            self.logger: logging.Logger = logging.getLogger(__name__)
            self.logger.disabled = True
            # self.logger.disabled = False
            # self.logger.addHandler(logging.StreamHandler(sys.stdout))
            # self.logger.setLevel(level=logging.DEBUG)
        else:
            self.logger: logging.Logger = logger.getChild("AIPlan4GridAgent")

    def reset(self, obs: BaseObservation):
        """
        This method is called at the beginning.
        It is implemented by agents to reset their internal state if needed.
        """
        self._prev_por_error.value[:] = 0.0
        conv_ = self.run_dc(obs)
        if conv_:
            self._prev_por_error.value[:] = self.flow_computed - obs.p_or
        else:
            self.logger.warning(
                "Impossible to intialize the agent"
                "because the DC powerflow did not converge."
            )

    def run_dc(self, actions: list, obs: BaseObservation):
        pass

    def _clean_vect(self, storage, redispatching):
        """Remove the value too small and set them at 0."""
        storage[np.abs(storage) < self.margin_sparse] = 0.0
        redispatching[np.abs(redispatching) < self.margin_sparse] = 0.0

    def _update_storage_power_obs(self, obs: BaseObservation):
        self._storage_power_obs.value += obs.storage_power.sum()

    def _update_inj_param(self, obs: BaseObservation):
        self._update_storage_power_obs(obs)

        self.load_per_bus.value[:] = 0.0
        self.gen_per_bus.value[:] = 0.0
        load_p = 1.0 * obs.load_p
        load_p *= (obs.gen_p.sum() - self._storage_power_obs.value) / load_p.sum()
        for bus_id in range(self.nb_max_bus):
            self.load_per_bus.value[bus_id] += load_p[
                self.bus_load.value == bus_id
            ].sum()
            self.gen_per_bus.value[bus_id] += obs.gen_p[
                self.bus_gen.value == bus_id
            ].sum()

    def _validate_param_values(self):
        self.storage_down._validate_value(self.storage_down.value)
        self.storage_up._validate_value(self.storage_up.value)

        self.redisp_up._validate_value(self.redisp_up.value)
        self.redisp_down._validate_value(self.redisp_down.value)

        self._th_lim_mw._validate_value(self._th_lim_mw.value)

        self._storage_target_bus._validate_value(self._storage_target_bus.value)
        self._past_dispatch._validate_value(self._past_dispatch.value)
        self._past_state_of_charge._validate_value(self._past_state_of_charge.value)

    def update_parameters(self, obs: BaseObservation):
        ## update the load / gen bus injected values
        self._update_inj_param(obs)
        # check that all parameters have correct values
        # for example non negative values for non negative parameters
        self._validate_param_values()

    def act(
        self, obs: BaseObservation, reward: float = 1.0, done: bool = False
    ) -> BaseAction:
        pass

    def step():
        pass
