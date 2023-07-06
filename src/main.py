import grid2op
from grid2op.Backend import PandaPowerBackend
from AIPlan4GridAgent import AIPlan4GridAgent


def main():
    env_name = "educ_case14_storage"
    env = grid2op.make(env_name, test=True, backend=PandaPowerBackend())
    horizon = 1
    data_generator = env.chronics_handler.real_data.data
    agent = AIPlan4GridAgent(
        env,
        horizon,
        data_generator,
    )
    env.set_id(0)
    obs = env.reset()
    agent.act(obs, reward=0, done=False)


if __name__ == "__main__":
    main()
