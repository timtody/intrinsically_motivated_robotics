import gym
import rlbench.gym
import utils
import numpy as np
from td3 import TD3
# from matplotlib import pyplot as plt

# env = gym.make('reach_target-state-v0')
# Alternatively, for vision:

ENV_NAME = 'reach_target-state-v0'
ENV_SEED = 42
MAX_TIMESTEPS = int(1e3)
START_TIME_STEPS = 1
EXPL_NOISE = 0.2


def eval_policy(policy, env_name, seed, eval_episodes=10):
    eval_env = gym.make(env_name)
    eval_env.seed(seed + 100)

    avg_reward = 0.
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        while not done:
            action = policy.select_action(np.array(state))
            state, reward, done, _ = eval_env.step(action)
            avg_reward += reward

    avg_reward /= eval_episodes

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    return avg_reward


if __name__ == "__main__":
    env = gym.make(ENV_NAME)
    print("done loading env")
    # env.seed(ENV_SEED)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "discount": 0.99,
        "tau": 0.005,
    }

    kwargs["policy_noise"] = 0.2 * max_action
    kwargs["noise_clip"] = 0.2 * max_action
    kwargs["policy_freq"] = 2
    policy = TD3(**kwargs)

    replay_buffer = utils.ReplayBuffer(state_dim, action_dim)

    # eval untrained policy
    # evaluations = [eval_policy(policy, ENV_NAME, ENV_SEED)]
    print("resetting env")
    state, done = env.reset(), False
    print("done resetting env")
    state = state

    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0
    for t in range(MAX_TIMESTEPS):
        episode_timesteps += 1

        if t < START_TIME_STEPS:
            action = env.action_space.sample()
        else:
            action = (policy.select_action(state) +
                      np.random.normal(0, max_action * EXPL_NOISE,
                                       size=action_dim)).clip(-max_action,
                                                              max_action)
        # perform action
        next_state, reward, done, _ = env.step(action)
        env.render()
        done = float(done) if episode_timesteps < MAX_TIMESTEPS else 1

        # Store data in replay buffer
        replay_buffer.add(state, action, next_state, reward, done)
        state = next_state
        episode_reward += reward

        if t >= START_TIME_STEPS:
            policy.train(replay_buffer, 100)

    print('Done')
    env.close()
