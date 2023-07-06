import grid2op
from grid2op.Backend import PandaPowerBackend
from AIPlan4GridAgent import AIPlan4GridAgent


def main():
    env_name = "educ_case14_storage"
    env = grid2op.make(env_name, test=True, backend=PandaPowerBackend())
    agent = AIPlan4GridAgent(
        env,
        horizon=144,
    )  # I think that the timestep duration is 5 minutes, so 144 timesteps represents 12 hours
    env.set_id(0)  # reset the env to the same id
    obs = env.reset()
    agent.act(obs, reward=0, done=False)


if __name__ == "__main__":
    main()
