from .experiment import BaseExperiment
import numpy as np
import torch
import pickle
import matplotlib.pyplot as plt


class Experiment(BaseExperiment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_steps = 100000
        self.n_steps = 1000000
        self.bsize = 5000

    def _gen_dataset(self):
        # first make the data set
        dataset = []
        state = self.env.reset()

        for i in range(self.n_steps):
            if i % 100000 == 0:
                print(f"rank {self.rank} at step", i)
            action = self.env.action_space.sample()
            next_state, *_ = self.env.step(action)
            dataset.append((state, next_state, action))
            state = next_state

        with open(f"out/iv_gen_dataset_rank{self.rank}.pt", "wb") as f:
            pickle.dump(dataset, f)

    def _test_iv(self):
        with open("out/dataset.p", "rb") as f:
            dataset = np.array(pickle.load(f))

        test_set = dataset[int(len(dataset) * 0.9) :]
        train_set = dataset[: int(len(dataset) * 0.9)]

        print("successfully loaded dataset of length", len(dataset))

        state = self.env.reset()

        for i in range(self.n_steps):
            print("training")
            idx = np.random.randint(len(test_set), size=self.bsize)
            state_batch, next_state_batch, action_batch = zip(*train_set[idx])
            loss = self.agent.icm.train_inverse(
                state_batch, next_state_batch, action_batch, eval=False
            )
            self.wandb.log({"training loss": loss.mean()})

            if i % 1000 == 0:
                print("evaluating...")
                state_batch, next_state_batch, action_batch = zip(*test_set)
                loss = self.agent.icm.train_inverse(
                    state_batch, next_state_batch, action_batch, eval=True
                )
                self.wandb.log({"eval loss": loss.mean()})

    def _train(self):
        print("Starting training...")
        dataset = []
        state = self.env.reset()

        for i in range(self.cnf.main.n_steps):
            action = self.env.action_space.sample()
            next_state, *_ = self.env.step(action)
            dataset.append((state, next_state, torch.tensor(action)))

            state = next_state

            if i % 250 == 249:
                print("training at step", i)
                state_batch, next_state_batch, action_batch = zip(*dataset)
                loss = self.agent.icm.train_inverse(
                    state_batch, next_state_batch, action_batch, eval=False
                )
                self.wandb.log({"batch loss": loss.mean()})
                dataset = []

                # self._test()
                state = self.env.reset()

    def _train_with_im(self):
        print("Starting training with im")
        dataset = []
        state = self.env.reset()

        for i in range(self.cnf.main.n_steps):
            action = self.agent.get_action(state)
            next_state, _, done, _ = self.env.step(action)

            trans = (state, next_state, action)
            dataset.append(trans)
            self.agent.append_icm_transition(*trans)

            state = next_state

            if i % 250 == 249:
                print("training at step", i)
                state_batch, next_state_batch, action_batch = zip(*dataset)
                loss = self.agent.icm.train_inverse(
                    state_batch, next_state_batch, action_batch, eval=False
                )
                self.wandb.log({"batch loss": loss.mean()})

                dataset = []
                done = True
                self.env.reset()

            self.agent.set_is_done(done)

            if done:
                results = self.agent.train()

    def _test(self):
        print("Starting testing...")
        state = self.env.reset()
        buffer = [state]
        predicted_actions = torch.zeros(50)
        action = [1] * self.cnf.env.action_dim
        true_actions = []

        for i in range(50):
            action = self.env.action_space.sample()
            true_actions.append(action)
            state, *_ = self.env.step(action)
            buffer.append(state)

        # get the predicted actions
        for i, (state, next_state) in enumerate(zip(buffer, buffer[1:])):
            predicted_actions[i] = self.agent.icm.get_action(state, next_state)

        true_actions = torch.tensor(true_actions).flatten()

        loss = torch.nn.functional.mse_loss(
            predicted_actions, true_actions, reduction="none"
        )

        plt.plot(loss.detach())

        self.wandb.log(
            {
                "test loss": loss.mean(),
                "loss per item": plt,
                "true actions": true_actions,
                "predicted actions": predicted_actions,
            }
        )
        plt.clf()

        # generalization loss
        predicted_actions = torch.zeros(len(buffer[1:]))
        for i, state in enumerate(buffer[1:]):
            predicted_actions[i] = self.agent.icm.get_action(buffer[0], state)

        generalization_loss = torch.nn.functional.mse_loss(
            predicted_actions, torch.ones(len(predicted_actions)), reduction="none"
        )

        plt.plot(generalization_loss.detach().numpy())
        self.wandb.log({"generalization loss": plt})
        plt.clf()

    def _reach_goal(self):
        # define goal
        state = self.env.reset()
        proto_action = [1] * self.cnf.env.action_dim

        for _ in range(70):
            state, *_ = self.env.step(proto_action)

        goal = state

        for _ in range(1000):
            state = self.env.reset()
            done = False
            episode_len = 0
            episode_reward = 0

            for _ in range(500):
                episode_len += 1
                inverse_action = self.agent.icm.get_action(state, goal)
                action = self.agent.get_action(state, inverse_action=inverse_action)
                state, reward, *_ = self.env.step(action)

                dist = ((state - goal) ** 2).sum()
                if dist < 1:
                    done = True
                    reward = 1
                else:
                    reward = -1

                episode_reward += reward

                self.agent.set_reward(reward)
                self.agent.set_is_done(done)

                if done:
                    break

            self.wandb.log(
                {"episode length": episode_len, "episode reward": episode_reward}
            )
            self.agent.train_ppo()

    def run(self):
        if self.cnf.main.with_im:
            self._train_with_im()
        else:
            self._train()
        self._reach_goal()
