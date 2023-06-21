from grid2op.Agent import BaseAgent
from grid2op.Environment import Environment
from grid2op.Action import BaseAction
from grid2op.Observation import BaseObservation
from typing import Optional
import logging
import numpy as np
import pandapower as pp
from pandapower.pd2ppc import _pd2ppc
from pandapower.pypower.makePTDF import makePTDF


class AIPlan4GridAgent(BaseAgent):
    def _get_base_actions(self):
        # redispatch actions
        _redispatch_actions = []
        for gen_id in range(len(self.env.storage_max_p_prod)):
            pmax = self.env.gen_pmax[gen_id]
            pmin = self.env.gen_pmin[gen_id]
            delta = int(pmax - pmin)
            for i in range(0, delta):
                _redispatch_actions.append((gen_id, i))
                _redispatch_actions.append((gen_id, -i))

        # storage actions
        _storage_actions = []
        for storage_id in range(self.env.n_storage):
            emax = self.env.storage_Emax[storage_id]
            emin = self.env.storage_Emin[storage_id]
            delta = int(emax - emin)
            for i in np.linspace(0, delta, delta * 10):
                _storage_actions.append((storage_id, round(i, 2)))
                _storage_actions.append((storage_id, -round(i, 2)))

        return _redispatch_actions, _storage_actions

    def _get_ptdf(self):
        net = self.env.backend._grid
        pp.rundcpp(net)
        _, ppci = _pd2ppc(net)
        ptdf = makePTDF(ppci["baseMVA"], ppci["bus"], ppci["branch"])
        return ptdf

    def __init__(
        self,
        env: Environment,
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

        self._redispatch_actions, self._storage_actions = self._get_base_actions()
        self._ptdf = self._get_ptdf()

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
        pass

    def step():
        pass
