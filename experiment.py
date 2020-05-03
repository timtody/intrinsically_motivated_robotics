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
    def __init__(self, cnf, rank, log=False, tb=True, mode=None):
        self.cnf = cnf
        self.log = log
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
        self.agent = PPO(self.action_dim, self.state_dim, **cnf.ppo)
        self.memory = Memory()

        # setup ICM
        self.icm = ICModule(cnf.env.action_dim, self.state_dim, **cnf.icm).to(
            self.device
        )
        self.icm_transition = collections.namedtuple(
            "icm_trans", ["state", "next_state", "action"]
        )
        self.icm_buffer = []

        # setup experiment variables
        self.global_step = 0
        self.ppo_timestep = 0

        # setup tensorboard
        if tb:
            if not cnf.main.train:
                self.writer = SummaryWriter(f"tb/mode:notrain_rank:{rank}")
            else:
                self.writer = SummaryWriter(f"tb/mode:{cnf.env.state}_rank:{rank}")

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


class CountCollisions(Experiment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # setup logging metrics
        self.n_collisions = 0
        self.n_sounds = 0
        self.reward_sum = 0

        # queues
        self.Q_LEN = self.cnf.main.train_each
        self.reward_Q = RewardQueue(self.Q_LEN, self.cnf.ppo.gamma)
        self.value_Q = ValueQueue(self.Q_LEN)
        self.value_buf = []

        # experiment parameters
        self.episode_len = 500

    def run(self, callbacks=[]):
        state = self.env.reset()
        results = defaultdict(lambda: 0)
        for i in range(self.cnf.main.n_steps):
            self.ppo_timestep += 1
            self.global_step += 1

            # env step
            if self.log and self.global_step % 5000 == 0:
                print("exp in mode", self.cnf.env.mode, "at step", self.global_step)

            if not self.cnf.main.train:
                action = self.env.action_space.sample()
            else:
                action, action_mean, entropy = self.agent.policy_old.act(
                    state, self.memory
                )

            next_state, _, done, info = self.env.step(action)

            # _headculate intrinsic reward
            # make icm trans
            trans = self.icm_transition(state, next_state, torch.tensor(action))
            # append to buffer
            self.icm_buffer.append(trans)

            # reset environment
            if self.global_step % self.episode_len == 0:
                done = True
                if self.cnf.main.train:
                    self.reset()

            if self.cnf.main.train:
                self.memory.is_terminals.append(done)

            self.value_buf.append(self.agent.get_value(next_state))

            # train agent
            if self.ppo_timestep % self.cnf.main.train_each == 0:

                # train agent
                state_batch, next_state_batch, action_batch = zip(*self.icm_buffer)
                self.icm_buffer = []

                im_loss = self.icm.train_forward(
                    state_batch, next_state_batch, action_batch
                )

                if self.cnf.main.train:
                    self.memory.rewards = im_loss.cpu().numpy()

                    ploss, vloss = self.agent.update(self.memory)
                    self.memory.clear_memory()
                    self.ppo_timestep = 0

                    self.writer.add_scalar("policy loss", ploss, self.global_step)
                    self.writer.add_scalar("value loss", vloss, self.global_step)

                self.reward_sum += im_loss.mean()

                self.writer.add_scalar(
                    "mean reward", self.reward_sum / self.global_step, self.global_step
                )
                self.writer.add_scalar(
                    "cumulative reward", self.reward_sum, self.global_step
                )

                # for s in range(self.cnf.main.train_each):
                #     self.reward_Q.push(im_loss[s])
                #     self.value_Q.push(self.value_buf[s])

                #     if s + self.global_step >= self.Q_LEN:
                #         step = self.global_step - self.cnf.main.train_each + s
                #         self.writer.add_scalars(
                #             "ret_approx", {
                #                 "true_ret": self.reward_Q,
                #                 "app_ret": self.value_Q
                #             }, step)

                self.value_buf = []

            state = next_state

            # receive callback info
            for i, cb in enumerate(callbacks):
                results[i] += cb(info)

            # retrieve metrics
            self.n_collisions += info["collided"]
            self.n_sounds += info["sound"]

            # log to tensorboard
            if self.cnf.main.train:
                # training-only metrics
                self.writer.add_histogram("action_mean", action_mean, self.global_step)
                self.writer.add_scalar("entropy", entropy, self.global_step)

            # rest of metrics
            # self.writer.add_scalar("reward", im_loss, self.global_step)

            self.writer.add_scalar("n_collisions", self.n_collisions, self.global_step)
            self.writer.add_scalar("n_sounds", self.n_sounds, self.global_step)

        self.env.close()
        return results.values()


class GoalReach(Experiment):
    def __init__(self, *args):
        super().__init__(*args)
        # init goal buffer here
        self.episode_len = 250

    def get_loss(self, state, goal):
        return ((state - goal) ** 2).mean()

    def run(self, log=False):
        # get a goal
        # goals = []
        print("generating goal")
        for i in range(100):
            goal, *_ = self.env.step([1] * self.cnf.env.action_dim)
        print("done.")

        for i in range(1000):
            self.global_step += 1
            self.ppo_timestep += 1

            state = self.env.reset()
            done = False
            episode_len = 0
            episode_reward = 0
            while not done:
                episode_len += 1
                if episode_len > self.episode_len:
                    done = True
                action, *_ = self.agent.policy_old.act(state, self.memory)
                next_state, *_ = self.env.step(action)

                reward = -self.get_loss(
                    self.icm.get_embedding(next_state), self.icm.get_embedding(goal),
                )
                if reward >= -0.001:
                    print("i've reached the goal")
                    reward += 1
                    done = True

                episode_reward += reward

                self.memory.rewards.append(reward)
                self.memory.is_terminals.append(done)

                if self.ppo_timestep % self.cnf.main.train_each == 0:
                    self.ppo_timestep = 0

                state = next_state

            self.agent.update(self.memory)
            self.memory.clear_memory()

            self.writer.add_scalar("episode reward", episode_reward, self.global_step)
            self.writer.add_scalar("episode len", episode_len, self.global_step)


class CheckActor(Experiment):
    """ Experiment to investigate the
    critic's function"""

    def run(self, callbacks, log=False):
        state = self.env.reset()
        results = defaultdict(lambda: 0)
        mean_reward = 0
        for i in range(self.cnf.main.n_steps):
            if log and i % 5000 == 0:
                print("exp in mode", self.cnf.env.mode, "at step", i)

            self.ppo_timestep += 1

            if not self.cnf.main.train:
                action = self.env.action_space.sample()
            else:
                action, action_mean, entropy = self.agent.policy_old.act(
                    state, self.memory
                )

            next_state, _, done, info = self.env.step(action)

            if self.cnf.main.train:
                im_loss = self.icm.train_forward(state, next_state, action)
                self.memory.rewards.append(im_loss)
                self.memory.is_terminals.append(done)
                mean_reward += im_loss
            state = next_state

            if self.cnf.main.train:
                if self.ppo_timestep % self.cnf.main.train_each == 0:
                    self.agent.update(self.memory)
                    self.memory.clear_memory()
                    self.ppo_timestep = 0

            # receive callback info
            for i, cb in enumerate(callbacks):
                results[i] += cb(info)

            # retrieve metrics
            self.n_collisions += info["collided"]

            # log to tensorboard
            if self.cnf.main.train:
                self.writer.add_scalar("reward", im_loss, self.global_step)
                self.writer.add_scalar(
                    "mean reward", mean_reward / self.global_step, self.global_step
                )

            self.global_step += 1
        self.env.close()
        return results.values()


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


class TestReward(Experiment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        torch.manual_seed(self.cnf.torch.seed)

        from utils import GraphWindow

        self.window = GraphWindow(
            ["reward"]
            + ["left_finger", "right_finger", "left_wrist", "right_wrist", "back"],
            6,
            1,
        )

    def run(self):
        self.window.update(0, 0, 0, 0, 0, 0)
        actions = [[0, 1, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0], [0, -1, 0, 0, 0, 0, 0]]
        actions = [torch.tensor(a) for a in actions]
        state = self.env.reset()
        reward_sum = 0
        for i in range(10000):
            print(i)
            self.agent.policy_old.act(state, self.memory)
            action = actions[0]
            if i > 50:
                action = actions[1]
            if i > 100:
                action = actions[2]
            if i > 180:
                action = actions[0]
            if i > 250:
                action = torch.tensor([0, 1, 0, 0, 0, 0, 0])

            next_state, *_ = self.env.step(action)
            reward = self.icm.train_forward([[state]], [[next_state]], [action])
            reward_sum += reward
            sensors = np.array(self.env.read_force_sensors_hand()).sum(axis=1)
            state = next_state
            self.window.update(reward, *sensors)
        print(reward_sum / 10000)


class Behavior(Experiment):
    def run(self):
        state = self.env.reset()
        for i in range(10000):
            self.ppo_timestep += 1

            action, *_ = self.agent.policy_old.act(state, self.memory)
            next_state, *_ = self.env.step(action)

            if self.cnf.main.train:
                im_loss = self.icm.train_forward([[state]], [[next_state]], [action])
                self.memory.rewards.append(im_loss)
                self.memory.is_terminals.append(False)

            if (
                self.ppo_timestep % self.cnf.main.train_each == 0
                and self.cnf.main.train
            ):
                print("Training")
                self.agent.update(self.memory)
                self.memory.clear_memory()

            state = next_state


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
            name=f"{self.cnf.wandb.name}_{kwargs['mode']}_rank{args[1]}",
            group=f"{self.cnf.wandb.name}_{kwargs['mode']}",
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
                    if "random_reward" in self.cnf.env.mode
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
        for i in range(1000000):
            action = self.env.action_space.sample()
            next_state, *_ = self.env.step(action)
            self.agent.append_icm_transition(state, next_state, torch.tensor(action))

            if i % 1000 == 999:
                results = self.agent.train(train_fw=True, train_ppo=False)
                self.wandb.log({"fw loss": results["imloss"].mean()}, step=i)

            state = next_state


class TestFWModelFromDB(Experiment):
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
        N_STEPS = 1000000
        
        with open("data/fwmodel_db.p", "rb") as f:
            db = pickle.load(f)
        
        for i in range(N_STEPS):
            pass


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

