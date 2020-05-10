from .experiment import BaseExperiment
import plotly.graph_objects as go
import numpy as np
import torch


class Experiment(BaseExperiment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # setup logging metrics
        self.n_collisions_self = 0
        self.n_collisions_other = 0
        self.n_collisions_dynamic = 0
        self.n_sounds = 0
        self.reward_sum = 0

        # experiment parameters
        self.episode_len = 500
        self.episode_reward = 0

        # gripper positions
        self.gripper_positions = []

        self.gripper_positions_means = []
        self.gripper_positions_acc = 0

        self.gripper_freq_0 = 10
        self.gripper_freq_1 = 1000

        self.pointcloud_every = 250000

        # running mean and std of reward
        self.running_mean = 0
        self.running_std = 0

    def make_pointclouds(self, step):
        gripx, gripy, gripz = zip(*self.gripper_positions)
        grip_mean_x, grip_mean_y, grip_mean_z = zip(*self.gripper_positions_means)

        cloud_high_freq = go.Figure(
            data=[
                go.Scatter3d(
                    x=gripx,
                    y=gripy,
                    z=gripz,
                    mode="markers",
                    marker=dict(
                        size=2,
                        color=np.arange(len(gripx)),
                        colorscale="Viridis",
                        opacity=0.8,
                    ),
                )
            ]
        )
        cloud_low_freq = go.Figure(
            data=[
                go.Scatter3d(
                    x=grip_mean_x,
                    y=grip_mean_y,
                    z=grip_mean_z,
                    mode="markers",
                    marker=dict(
                        size=8,
                        color=np.arange(len(grip_mean_x)),
                        colorscale="Viridis",
                        opacity=0.8,
                    ),
                )
            ]
        )
        cloud_high_freq.write_html(
            f"data/pointclouds/{self.cnf.wandb.name}_high_freq_{self.rank}_{step}.html"
        )
        cloud_low_freq.write_html(
            f"data/pointclouds/{self.cnf.wandb.name}_low_freq_rank_{self.rank}_{step}.html"
        )

        # gripper positions
        self.gripper_positions = []

        self.gripper_positions_means = []
        self.gripper_positions_acc = 0

    def run(self):
        state = self.env.reset()
        for i in range(self.cnf.main.n_steps):
            self.ppo_timestep += 1
            self.global_step += 1

            # record pointcloud stuff
            if i % self.gripper_freq_0 == 0:
                self.gripper_positions.append(self.env._gripper.get_position())

            if i % self.gripper_freq_1 == self.gripper_freq_1 - 1:
                self.gripper_positions_means.append(
                    self.gripper_positions_acc / self.gripper_freq_1
                )
                self.gripper_positions_acc = 0

            self.gripper_positions_acc += np.array(self.env._gripper.get_position())

            # save the clouds
            if i % self.pointcloud_every == self.pointcloud_every - 1:
                self.make_pointclouds(i)

            if not self.cnf.main.train:
                action = self.env.action_space.sample()
            else:
                action = self.agent.get_action(state)

            next_state, _, done, info = self.env.step(action)

            self.agent.append_icm_transition(state, next_state, torch.tensor(action))

            # reset environment
            if self.global_step % self.episode_len == 0:
                done = True
                # -------------
                self.env.reset()
                # TODO: change back
                # if self.cnf.main.train:
                #     self.env.reset()

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

                # compute running mean and std
                prev_mean = self.running_mean
                self.running_mean = (
                    self.running_mean
                    + (batch_reward - self.running_mean) / self.global_step
                )
                self.running_std = self.running_std + (
                    batch_reward - self.running_mean
                ) * (batch_reward - prev_mean)

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

        self.wandb.log(
            {
                "running mean": self.running_mean,
                "running std": np.sqrt(self.running_std / self.global_step),
            }
        )
        self.env.close()

        return self.n_collisions_self, self.reward_sum
