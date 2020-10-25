import torch
import numpy as np
import pandas as pd
from .experiment import BaseExperiment


class Experiment(BaseExperiment):
    """
    Visualise the performance of the forward model in joint angle space
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def load_dataset(self):
        dataset = torch.load(f"results/forward/dataset/database_{self.cnf.env.state}")
        split = 0.95
        dataset = np.array(dataset)
        np.random.shuffle(dataset)
        test_set = dataset[int(len(dataset) * split) :]
        train_set = dataset[: int(len(dataset) * split)]
        return train_set, test_set

    def run(self, pre_run):
        eval_each = 10
        train_set, test_set = np.array(self.load_dataset())
        bsize = 256
        idx = np.random.randint(len(train_set), size=bsize)
        for i in range(self.cnf.main.n_steps):
            state, nstate, action = zip(*train_set[idx])
            # oh lord help me
            action = [torch.tensor(ac) for ac in action]
            loss = self.agent.icm.train_forward(state, nstate, action)

            if i % eval_each == eval_each - 1:
                state, nstate, action = zip(*test_set)
                state, nstate, action = zip(*train_set[idx])
                # oh lord help me
                action = [torch.tensor(ac) for ac in action]
                loss = self.agent.icm.train_forward(state, nstate, action)

        state = self.env.reset()

        # get the reward for the touch event
        for i in range(50):
            self.env.step([0, 1, 0, 0, 0, 0, 0])
        exit(1)

        return results

    @staticmethod
    def plot(results, cnf):
        res = []
        for key, value in results.items():
            res += value
        df = pd.DataFrame(
            data=res, columns=["Rank", "State", "Batch update", "Eval loss"]
        )
        df.to_csv(f"results/forward/ms1/{cnf.env.state}_res.csv")

