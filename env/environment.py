import os
import gym
import numpy as np
from pyrep import PyRep
from pyrep.robots.arms.panda import Panda
from pyrep.robots.end_effectors.panda_gripper import PandaGripper
from pyrep.objects.shape import Shape
from observation import Observation


class Env(gym.Env):
    def __init__(self, cnf):
        self.cnf = cnf.env
        self._launch()
        self._setup_robot()
        self._setup_shapes()
        self._set_objects_collidable()

        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(7,)
        )
        # TODO: need to be made more general for vision space
        obs = self.step([0, 0, 0, 0, 0, 0, 0])
        obs = self.reset()
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=obs.get_all().shape
        )

    def _launch(self):
        scene_path = os.path.abspath(os.path.join(
            'scenes', self.cnf.scene_path))
        self.pr = PyRep()
        self.pr.launch(scene_path, headless=self.cnf.headless)
        self.pr.start()

    def _setup_robot(self):
        self.arm = Panda()
        self.arm.set_control_loop_enabled(False)
        self.arm.set_motor_locked_at_zero_velocity(True)
        self.gripper = PandaGripper()

    def _setup_shapes(self):
        self.table = Shape("diningTable_visible")

    def _set_vels(self, action):
        self.arm.set_joint_target_velocities(action)

    def _get_vision(self):
        # TODO: implement
        pass

    def _get_reward(self):
        # TODO: implement
        return 0

    def _get_done(self):
        # TODO: implement
        return False

    def _get_info(self):
        # TODO: maybe implement (more feats)
        info = dict(
            collided_with_table=self.check_collision_with_table()
        )
        return info

    def _get_observation(self):
        obs = Observation(
            self.arm.get_joint_velocities(),
            self.arm.get_joint_positions(),
            self.arm.get_joint_forces(),
            self.gripper.get_open_amount(),
            self.gripper.get_pose(),
            self.gripper.get_joint_positions(),
            self.gripper.get_touch_sensor_forces(),
            self._get_vision() if self.cnf.state == "vision" else None
        )
        return obs

    def _set_objects_collidable(self):
        self.arm.set_collidable(True)
        self.gripper.set_collidable(True)
        self.table.set_collidable(True)

    def step(self, action):
        # TODO: do something more with action
        self._set_vels(action)
        self.pr.step()
        obs = self._get_observation()
        reward = self._get_reward()
        done = self._get_done()
        info = self._get_info()
        return obs, reward, done, info

    def reset(self):
        # TODO: implement reset (not important for now)
        return self._get_observation()

    def render(self):
        # TODO: refer to RLBench gym implementation
        pass

    def close(self):
        self.pr.stop()
        self.pr.shutdown()

    def check_collision_with_table(self):
        return self.table.check_collision_by_handle(
            self.arm._collision_collection)
