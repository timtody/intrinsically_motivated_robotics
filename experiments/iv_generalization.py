from .experiment import BaseExperiment
import numpy as np
import torch
import pickle


class Experiment(BaseExperiment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_steps = 10000000
        self.bsize = 1000

    def _gen_dataset(self):
        # first make the data set
        dataset = []
        state = self.env.reset()

        for i in range(self.n_steps):
            action = self.env.action_space.sample()
            next_state, *_ = self.env.step(action)
            dataset.append((state, next_state, action))
            state = next_state

        with open(f"out/iv_gen_dataset_rank{self.rank}.pt", "wb") as f:
            pickle.dump(dataset, f)

    def run(self):
        with open("out/dataset.p", "rb") as f:
            dataset = np.array(pickle.load(f))

        test_set = dataset[int(len(dataset) * 0.9) :]
        train_set = dataset[: int(len(dataset) * 0.9)]

        print("successfully loaded dataset of length", len(dataset))

        state = self.env.reset()

        for i in range(self.n_steps):
            idx = np.random.randint(len(test_set), size=self.bsize)
            state_batch, next_state_batch, action_batch = zip(*train_set[idx])
            loss = self.agent.icm.train_inverse(
                state_batch, next_state_batch, action_batch, eval=False
            )
            self.wandb.log({"training loss": loss.sum()})

            if i % 100000 == 0:
                print("evaluating...")
                state_batch, next_state_batch, action_batch = zip(*test_set)
                loss = self.agent.icm.train_inverse(
                    state_batch, next_state_batch, action_batch, eval=True
                )
                self.wandb.log({"eval loss": loss.sum()})
