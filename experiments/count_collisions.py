from .experiment import BaseExperiment
import plotly.graph_objects as go
import numpy as np
import torch
import time
import os


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

        # watch models
        self.wandb.watch(self.agent.icm)
        self.wandb.watch(self.agent.ppo.policy)
        self.wandb.watch(self.agent.ppo.policy_old)

        if self.cnf.main.checkpoint:
            self.load_state(
                os.path.join(self.cnf.main.checkpoint, "rank" + str(self.rank))
            )

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

    def get_joint_entropies(self, joint_angles, joint_intervals):
        """
        This function makes an estimate of the distribution of joint angles
        per joint by using the maximum likelihood estimate to fit a
        categorical distribution to the data. Then, entropies per
        distribution/joint are computed and returned.
        This is done to asses how the range of motion per joint evolves during
        experiments.
        """
        entropies = []
        for i, angles in enumerate(joint_angles):
            hist = torch.histc(
                torch.tensor(angles),
                min=joint_intervals[i][0],
                max=joint_intervals[i][1],
            )
            c = torch.distributions.categorical.Categorical(logits=hist)
            entropies.append(c.entropy())
        return entropies

    def get_joint_min_max(self, joint_angles):
        min_max = []
        for angle in joint_angles:
            min_max.append([min(angle), max(angle)])
        return min_max

    def get_joint_range_in_percent(self, joint_angles, joint_intervals):
        ranges = []
        for i, angles in enumerate(joint_angles):
            dist = max(angles) - min(angles)
            max_dist = max(joint_intervals[i]) - min(joint_intervals[i])
            ranges.append(dist / (max_dist + 0.0001))
        return ranges

    def _make_pointclouds(self, step):
        # record pointcloud stuff
        if step % self.gripper_freq_0 == 0:
            self.gripper_positions.append(self.env._gripper.get_position())

        if step % self.gripper_freq_1 == self.gripper_freq_1 - 1:
            self.gripper_positions_means.append(
                self.gripper_positions_acc / self.gripper_freq_1
            )
            self.gripper_positions_acc = 0

        self.gripper_positions_acc += np.array(self.env._gripper.get_position())

        # save the clouds
        if step % self.pointcloud_every == self.pointcloud_every - 1:
            self.make_pointclouds(step)

    def run(self):
        joint_intervals = self.env.get_joint_intervals()
        joint_angles = []
        actions_norms = []
        state = self.env.reset()
        for i in range(self.cnf.main.n_steps):
            self_collisions_batch = 0
            ext_collisions_batch = 0
            dyn_collisions_batch = 0

            it_start = time.time()
            self.ppo_timestep += 1
            self.global_step += 1

            if self.cnf.main.record_pc:
                self._make_pointclouds()

            if not self.cnf.main.train:
                action = torch.tensor(self.env.action_space.sample())
            else:
                action = self.agent.get_action(state)

            next_state, _, done, info = self.env.step(action)

            self.agent.append_icm_transition(state, next_state, action)

            # reset environment
            if self.global_step % self.episode_len == 0:
                if self.cnf.main.reset:
                    done = True
                    self.env.reset()

            self.agent.set_is_done(done)

            # retrieve metrics
            self.n_collisions_self += info["collided_self"]
            self.n_collisions_other += info["collided_other"]
            self.n_collisions_dynamic += info["collided_dyn"]
            self_collisions_batch += info["collided_self"]
            ext_collisions_batch += info["collided_other"]
            dyn_collisions_batch += info["collided_dyn"]

            # TODO: reintroduce this
            # self.n_sounds += info["sound"]

            # accumulate joint angles
            joint_angles.append(self.env.get_joint_angles())
            actions_norms.append(action.norm())

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

                ## general movements stuff

                # cumpute joint angle histograms
                # this is done to assess the range of motion
                joint_angles = list(zip(*joint_angles))
                # joint_min_max = self.get_joint_min_max(joint_angles)
                joint_ranges = self.get_joint_range_in_percent(
                    joint_angles, joint_intervals
                )
                joint_entropies = self.get_joint_entropies(
                    joint_angles, joint_intervals
                )
                joint_entropies_mean = sum(joint_entropies) / 7
                joint_ranges_mean = sum(joint_ranges) / 7

                # compute the mean norm of the actions
                actions_norms = torch.tensor(actions_norms).mean()

                # if we don't train we still want to log all the relevant data
                self.wandb.log(
                    {
                        "n collisions self": self.n_collisions_self,
                        "n collisions other": self.n_collisions_other,
                        "n collisions dyn": self.n_collisions_dynamic,
                        "col rate self": self.n_collisions_self / self.global_step,
                        "col rate other": self.n_collisions_other / self.global_step,
                        "col rate dyn": self.n_collisions_dynamic / self.global_step,
                        "batch col self": self_collisions_batch,
                        "batch col ext": ext_collisions_batch,
                        "batch col dyn": dyn_collisions_batch,
                        # "n sounds": self.n_sounds,
                        "cum reward": self.reward_sum,
                        "batch reward": batch_reward,
                        "policy loss": train_results["ploss"],
                        "value loss": train_results["vloss"],
                        "iteration time": time.time() - it_start,
                        "joint entropy mean": joint_entropies_mean,
                        "joint ranges mean": joint_ranges_mean,
                        "mean action norm": actions_norms,
                        **{
                            f"joint {i} ent": ent
                            for i, ent in enumerate(joint_entropies)
                        },
                        # **{f"joint {i} min": min_max[0] for min_max in joint_min_max},
                        # **{f"joint {i} max": min_max[1] for min_max in joint_min_max},
                        **{
                            f"joint {i} range": joint_range
                            for i, joint_range in enumerate(joint_ranges)
                        },
                    },
                    step=self.global_step,
                )
                joint_angles = []
                actions_norms = []

            state = next_state

            # save state
            if (
                self.global_step % self.cnf.main.save_every
                == self.cnf.main.save_every - 1
                and self.cnf.main.save_state
            ):
                self.save_state()

        self.wandb.log(
            {
                "running mean": self.running_mean,
                "running std": np.sqrt(self.running_std / self.global_step),
            }
        )
        self.env.close()

    def save_state(self):
        print("saving state to", self.checkpoint_path)
        cp_path = os.path.join(
            self.checkpoint_path, str(self.global_step), "rank" + str(self.rank),
        )
        if not os.path.exists(cp_path):
            os.makedirs(cp_path)

        state = {
            "global_step": self.global_step,
            "reward_sum": self.reward_sum,
        }
        torch.save(state, os.path.join(cp_path, "exp_state.p"))
        self.agent.save_state(cp_path)

    def load_state(self, path):
        print("loading state from", path)
        exp_state = torch.load(os.path.join(path, "exp_state.p"))
        self.global_step = exp_state["global_step"]
        self.reward_sum = exp_state["reward_sum"]
        self.agent.load_state(path)
