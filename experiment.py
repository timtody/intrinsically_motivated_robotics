import os
import torch
import pickle
import time
import json
import wandb
import collections
import numpy as np
from observation import Observation
from env.environment import Env
from agent import Agent
from utils import RewardQueue, ValueQueue
from algo.ppo_cont import PPO, Memory
from algo.models import ICModule
from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict
from abc import abstractmethod


class Experiment:
    def __init__(self, cnf, rank):
        self.cnf = cnf
        self.rank = rank

        # set random seeds
        np.random.seed(cnf.env.np_seed)
        torch.manual_seed(cnf.env.torch_seed)

        # setup env
        env = Env(cnf)
        # skip_wrapper = SkipWrapper(cnf.env.skip)
        # self.env = skip_wrapper(env)
        self.env = env

        # pytorch device
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        # setup agent
        self.action_dim = cnf.env.action_dim
        self.state_dim = env.observation_space.shape[0]

        self.agent = Agent(self.action_dim, self.state_dim, self.cnf, self.device)

        # setup experiment variables
        self.global_step = 0
        self.ppo_timestep = 0

        self.wandb = wandb

        self.wandb.init(
            config=self.cnf,
            project=self.cnf.wandb.project,
            name=f"{self.cnf.wandb.name}_rank{self.rank}",
            group=f"{self.cnf.wandb.name}",
        )

    @abstractmethod
    def run(self, callbacks, log=False):
        pass

    def reset(self):
        for _ in range(100):
            self.env.step(self.env.action_space.sample() * 10)

    def save_state(self, step, additional_state={}):
        path = os.path.join("checkpoints", str(step))
        if not os.path.exists(path):
            os.mkdir(path)
        self._save_extra(step, additional_state)
        self._save(step)
        self.agent.save_state(step)
        self.env.save_state(step)

    def load_state(self, checkpoint, load_env=False) -> dict:
        extra_state = {}  # self._load_extra(checkpoint)
        self._load(checkpoint)
        self.agent.load_state(checkpoint)
        self.env.load_state(checkpoint)
        return extra_state

    def _save_extra(self, step, state) -> None:
        save_path = os.path.join("checkpoints", str(step), "extra_state.p")
        with open(save_path, "wb") as f:
            pickle.dump(state, f)

    def _load_extra(self, checkpoint) -> dict:
        abspath = os.environ["owd"]
        load_path = os.path.join(abspath, checkpoint, "exp_state.json")
        with open(load_path, "rb") as f:
            extra_state = pickle.load(f)
        return extra_state

    def _save(self, step) -> None:
        save_path = os.path.join("checkpoints", str(step), "exp_state.json")
        print("saving exp at", save_path)
        state = dict(
            global_step=self.global_step,
            ppo_step=self.ppo_timestep,
            icm_buffer=self.icm_buffer,
            torch_seed=self.cnf.env.torch_seed,
            np_seed=self.cnf.env.np_seed,
        )
        with open(save_path, "w") as f:
            json.dump(state, f)

    def _load(self, checkpoint) -> None:
        # restores the state of the experiment
        abspath = os.environ["owd"]
        load_path = os.path.join(abspath, checkpoint, "exp_state.json")
        print("loading exp state from", load_path)
        with open(load_path, "r") as f:
            state = json.load(f)
        self.global_step = state["global_step"]
        self.ppo_timestep = state["ppo_step"]
        self.icm_buffer = state["icm_buffer"]
        # set random seeds
        torch.manual_seed(state["torch_seed"])
        np.random.seed(state["np_seed"])


class NormalizeObs(Experiment):
    def run(self):
        tac = []
        prop = []
        audio = []

        for i in range(100000):
            obs, *_ = self.env.step(self.env.action_space.sample())
            tac.append(obs.get_filtered("touch"))
            prop.append(obs.get_filtered("joint"))
            audio.append(obs.get_audio())

        tac = torch.tensor(tac).float()
        prop = torch.tensor(prop).float()
        audio = torch.tensor(audio).float()

        print(f"tac:\n\tmean:{tac.mean()}\n\tstd:{tac.std()}")
        print(f"prop:\n\tmean:{prop.mean()}\n\tstd:{prop.std()}")
        print(f"audio:\n\tmean:{audio.mean()}\n\tstd:{audio.std()}")


class GoalReachAgent(Experiment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.agent = Agent(self.action_dim, self.state_dim, self.cnf, self.device)
        from utils import GraphWindow

        # experiment parameters
        self.episode_len = 250
        # self.win = GraphWindow(["reward"], 1, 1)

    def get_loss(self, state, goal):
        return ((state - goal) ** 2).mean()

    def run(self):
        state = self.env.reset()

        print("generating goal")
        action = [1, 1]
        for i in range(100):
            goal, *_ = self.env.step([1] * self.cnf.env.action_dim)
            # goal, *_ = self.env.step(action)
        print("done.")

        checkpoint = self.cnf.log.checkpoint
        if checkpoint:
            print("loading exp checkpoint")
            self.load_state(checkpoint)

        for i in range(1000):
            self.global_step += 1

            state = self.env.reset()
            done = False
            episode_len = 0
            episode_reward = 0
            while not done:
                episode_len += 1
                if episode_len > self.episode_len:
                    done = True
                action = self.agent.get_action(state)
                next_state, *_ = self.env.step(action)

                reward = -self.get_loss(
                    self.icm.get_embedding(next_state), self.icm.get_embedding(goal),
                )
                if reward >= -0.001:
                    print("i've reached the goal")
                    reward += 1
                    done = True

                episode_reward += reward

                self.agent.set_reward(reward)
                self.agent.set_is_done(done)

                state = next_state

            self.agent.ppo.update(self.agent.ppo_mem)
            self.agent.ppo_mem.clear_memory()

            self.writer.add_scalar("episode reward", episode_reward, self.global_step)
            self.writer.add_scalar("episode len", episode_len, self.global_step)

            if i % 100 == 0:
                self.save_state(i)

        print("saving")


class CountCollisionsAgent(Experiment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        import wandb

        self.wandb = wandb

        self.wandb.init(
            config=self.cnf,
            project=self.cnf.wandb.project,
            name=f"{self.cnf.wandb.name}_{self.cnf.env.state}_rank{args[1]}",
            group=f"{self.cnf.wandb.name}_{self.cnf.env.state}",
        )

        self.agent = Agent(self.action_dim, self.state_dim, self.cnf, self.device)

        # setup logging metrics
        self.n_collisions_self = 0
        self.n_collisions_other = 0
        self.n_collisions_dynamic = 0
        self.n_sounds = 0
        self.reward_sum = 0

        # experiment parameters
        self.episode_len = 500
        self.episode_reward = 0

    def run(self):
        obs = []
        state = self.env.reset()
        for i in range(self.cnf.main.n_steps):
            self.ppo_timestep += 1
            self.global_step += 1

            # env step
            if self.log and self.global_step % 5000 == 0:
                print("exp in mode", self.cnf.env.mode, "at step", self.global_step)

            if not self.cnf.main.train:
                action = self.env.action_space.sample()
            else:
                action = self.agent.get_action(state)

            next_state, _, done, info = self.env.step(action)

            self.agent.append_icm_transition(state, next_state, torch.tensor(action))

            # reset environment
            if self.global_step % self.episode_len == 0:
                done = True
                if self.cnf.main.train:
                    self.env.reset()

            self.agent.set_is_done(done)

            # retrieve metrics
            self.n_collisions_self += info["collided_self"]
            self.n_collisions_other += info["collided_other"]
            self.n_collisions_dynamic += info["collided_dyn"]

            # TODO: reintroduce this
            # self.n_sounds += info["sound"]

            # train agent
            if self.ppo_timestep % self.cnf.main.train_each == 0:
                # train and log resulting metrics
                train_results = self.agent.train(
                    train_ppo=self.cnf.main.train,
                    random_reward=True
                    if "random_reward" in self.cnf.env.state
                    else False,
                )

                batch_reward = train_results["imloss"].sum().item()
                self.reward_sum += batch_reward

                # if we don't train we still want to log all the relevant data
                self.wandb.log(
                    {
                        "n collisions self": self.n_collisions_self,
                        "n collisions other": self.n_collisions_other,
                        "n collisions dyn": self.n_collisions_dynamic,
                        "col rate self": self.n_collisions_self / self.global_step,
                        "col rate other": self.n_collisions_other / self.global_step,
                        "col rate dyn": self.n_collisions_dynamic / self.global_step,
                        "n sounds": self.n_sounds,
                        "cum reward": self.reward_sum,
                        "batch reward": batch_reward,
                        "policy loss": train_results["ploss"],
                        "value loss": train_results["vloss"],
                    },
                    step=self.global_step,
                )

            state = next_state
            obs.append(state)
        obs = np.concatenate(obs)

        print(obs.mean(), obs.std())
        self.env.close()
        return self.n_collisions_self, self.reward_sum


class MeasureForgetting(Experiment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        import wandb

        self.wandb = wandb

        self.wandb.init(
            config=self.cnf,
            project=self.cnf.wandb.project,
            name=f"{self.cnf.wandb.name}_rank{args[1]}",
            group=f"{self.cnf.wandb.name}",
        )

        self.agent = Agent(self.action_dim, self.state_dim, self.cnf, self.device)

        # setup logging metrics
        self.n_collisions_self = 0
        self.n_collisions_other = 0
        self.n_collisions_dynamic = 0
        self.n_sounds = 0
        self.reward_sum = 0

        # experiment parameters
        self.episode_len = 250

        self.burnin_len = 250000
        self.buffer_length = 250
        self.buffer_gap = 12

        self.train_len = 500000
        self.train_every = 500

        # buffer for transitions
        self.trans_buffer = []

        self.global_step = 0

    def _burn_in(self):
        state = self.env.reset()
        print("starting burn-in")
        for i in range(self.burnin_len):

            action = self.agent.get_action(state)
            next_state, _, done, info = self.env.step(action)
            self.agent.append_icm_transition(state, next_state, action)

            if i % self.episode_len == self.episode_len - 1:
                done = True
                self.env.reset()

            self.agent.set_is_done(done)

            if i % self.train_every == self.train_every - 1:
                # print("training")
                self.agent.train()

            state = next_state

    def _gather_dataset(self):
        # collect data
        print("start gathering data...")
        state = self.env.reset()
        for i in range(self.buffer_length * self.buffer_gap):
            action = self.agent.get_action(state)
            next_state, _, done, info = self.env.step(action)

            if i % self.buffer_gap == 0:
                self.trans_buffer.append([state, next_state, action])

            state = next_state

    def _create_db(self):
        with open(f"rank_{self.rank}_database.p", "wb") as f:
            pickle.dump(self.trans_buffer, f)
        self.agent.save_state(f"rank_{self.rank}_")

    def _restore_exp(self):
        # restore saved database of transitions
        with open(f"data/forget/{self.rank}/database.p", "rb") as f:
            self.trans_buffer = pickle.load(f)
        # restore agent
        self.agent.load_state(f"data/forget/{self.rank}")

    def _test_forgetting(self):

        # start the exp
        self.agent.reset_buffers()
        state = self.env.reset()
        print("starting training")
        for i in range(self.train_len):
            self.global_step += 1
            action = self.agent.get_action(state)
            next_state, _, done, info = self.env.step(action)
            self.agent.append_icm_transition(state, next_state, action)

            if i % self.episode_len == self.episode_len - 1:
                done = True
                self.env.reset()

            self.agent.set_is_done(done)

            if i % self.train_every == self.train_every - 1:
                # print("training")
                # train the agent
                self.agent.train()
                # check how performance has changed after training
                state_batch, next_state_batch, action_batch = zip(*self.trans_buffer)
                loss = self.agent.icm.train_forward(
                    state_batch, next_state_batch, action_batch, freeze=True
                )
                wandb.log({"mean loss": loss.mean()}, step=self.global_step)

            state = next_state

    def run(self):
        self._burn_in()
        self._gather_dataset()
        self._create_db()
        self._test_forgetting()


class TestFWModel(Experiment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        import wandb

        self.wandb = wandb

        self.wandb.init(
            config=self.cnf,
            project=self.cnf.wandb.project,
            name=f"{self.cnf.wandb.name}_rank{args[1]}",
            group=f"{self.cnf.wandb.name}",
        )

        self.agent = Agent(self.action_dim, self.state_dim, self.cnf, self.device)

    def run(self):
        state = self.env.reset()
        for i in range(self.cnf.main.n_steps):
            action = self.env.action_space.sample()
            next_state, *_ = self.env.step(action)
            self.agent.append_icm_transition(state, next_state, torch.tensor(action))

            if i % self.cnf.main.train_each == self.cnf.main.train_each - 1:
                results = self.agent.train(train_fw=True, train_ppo=False)
                self.wandb.log({"fw loss": results["imloss"].mean()}, step=i)

            state = next_state


class TestFWModelFromDB(Experiment):
    def run(self):
        N_STEPS = 1000000
        BATCH_SIZE = 5000

        with open("data/fwmodel_db.p", "rb") as f:
            db = pickle.load(f)
        db = np.array(db)
        train_set = db[:90000]
        test_set = db[90000:]

        for i in range(N_STEPS):
            idx = np.random.randint(len(train_set), size=BATCH_SIZE)
            state_batch, next_state_batch, action_batch = zip(*train_set[idx])

            # train fw model
            train_loss = self.agent.icm.train_forward(
                state_batch, next_state_batch, torch.tensor(action_batch),
            )
            wandb.log({"training loss": train_loss.mean().cpu()})
            if i % 5000 == 0:
                # test
                idx = np.random.randint(len(test_set), size=BATCH_SIZE)
                state_batch, next_state_batch, action_batch = zip(*test_set[idx])
                test_loss = self.agent.icm.train_forward(
                    state_batch, next_state_batch, torch.tensor(action_batch),
                )
                wandb.log({"test loss": test_loss.mean().cpu()})


class TestStateDifference(Experiment):
    def run(self):
        state = self.env.reset()
        diffs = []

        for i in range(1000):
            action = self.env.action_space.sample()
            next_state, *_ = self.env.step(action)

            state = torch.tensor(state).float().to(self.device)
            next_state = torch.tensor(next_state).float().to(self.device)

            diffs.append(torch.nn.functional.mse_loss(state, next_state,).mean())

            state = next_state

        stacked_diffs = torch.stack(diffs)

        print("diff mean:", stacked_diffs.mean())
        print("diff std:", stacked_diffs.std())


class CreateFWDB(Experiment):
    def run(self):
        db = []

        DB_SIZE = 100000

        state = self.env.reset()
        for i in range(DB_SIZE):
            action = self.env.action_space.sample()
            next_state, *_ = self.env.step(action)
            db.append([state, next_state, action])

        with open("data/fwmodel_db.p", "wb") as f:
            pickle.dump(db, f)


class GoalBasedExp(Experiment):
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

        exit(1)

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
