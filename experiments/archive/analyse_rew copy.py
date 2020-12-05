import torch
import numpy as np
import pandas as pd
from .experiment import BaseExperiment


class Experiment(BaseExperiment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _run(self):
        state = self.env.reset()
        results = []
        batch = []
        train_each = 100
        for i in range(self.cnf.main.n_steps):
            action = self.env.action_space.sample()
            next_state, *_ = self.env.step(action)
            action = torch.tensor(action)
            batch.append((state, next_state, action))
            state = next_state

            if i % train_each == train_each - 1:
                state_batch, next_state_batch, action_batch = zip(*batch)

                reward = self.agent.icm.train_forward(
                    state_batch, next_state_batch, action_batch,
                )
                batch = []
                results += reward.cpu().numpy().tolist()
        return results

    def run(self, pre_run):
        results = []
        cyclic, limits = self.env._arm.get_joint_intervals()
        old_limits = np.array(limits)
        constrained_limits = old_limits * 0.05
        limits_a = constrained_limits
        limits_a[1] += 1
        limits_b = limits_a.copy()
        limits_b[0] += -1
        limits_c = limits_b.copy()
        limits_c[0] += 1
        # set up constrained limits first
        self.env._arm.set_joint_intervals(cyclic, limits_a.tolist())
        res_a = self._run()
        self.env._arm.set_joint_intervals(cyclic, limits_b.tolist())
        res_b = self._run()
        self.env._arm.set_joint_intervals(cyclic, limits_c.tolist())
        res_c = self._run()
        # set up constrained limits first
        self.env._arm.set_joint_intervals(cyclic, limits_a.tolist())
        res_d = self._run()

        return res_a + res_b + res_c + res_d

    @staticmethod
    def plot(results, cnf):
        # print(results)
        df = pd.DataFrame(data=dict(results))
        df.to_csv("results/forward/ms0/res_test_back.csv")
