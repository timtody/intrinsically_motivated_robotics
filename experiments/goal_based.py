from .experiment import BaseExperiment
from agent import Agent
import numpy as np


class Experiment(BaseExperiment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.agent = Agent(self.action_dim, self.state_dim * 2, self.cnf, self.device)

    def compute_reward(self, state, goal):
        return -((goal - state) ** 2).sum()

    def run(self):
        goal_buffer = []

        # generate goals
        self.env.reset()
        for j in range(50):
            state, *_ = self.env.step([0, 0, 0, -1, 0, 0, 0])
        goal_buffer.append(state)

        self.env.reset()
        for j in range(40):
            state, *_ = self.env.step([0, 1, 0, 0, 0, 0, 0])
        goal_buffer.append(state)

        self.env.reset()
        for j in range(40):
            state, *_ = self.env.step([1, 1, 0, 0, 0, 0, 0])
        goal_buffer.append(state)

        self.env.reset()
        for j in range(40):
            state, *_ = self.env.step([-1, 1, 0, 0, 0, 0, 0])
        goal_buffer.append(state)

        self.env.reset()
        for j in range(40):
            state, *_ = self.env.step([0.5, 1, 0, 0, 0, 0, 0])
        goal_buffer.append(state)

        self.env.reset()
        for j in range(40):
            state, *_ = self.env.step([-0.5, 1, 0, 0, 0, 0, 0])
        goal_buffer.append(state)

        for i in range(100000):
            # pick goal
            goal = goal_buffer[np.random.randint(len(goal_buffer))]
            state = self.env.reset()
            episode_len = 0
            episode_reward = 0
            for j in range(250):
                episode_len += 1
                action = self.agent.get_action(np.concatenate([goal, state]))
                state, _, done, _ = self.env.step(action)
                reward = self.compute_reward(state, goal_buffer[0])

                episode_reward += reward
                if reward > -0.2:
                    done = True

                self.agent.set_reward(reward)
                self.agent.set_is_done(done)

                if done:
                    break

            self.wandb.log(
                {"episode length": episode_len, "episode reward": episode_reward}
            )

            self.agent.train_ppo()
