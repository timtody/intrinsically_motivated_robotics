import os
import gym
import numpy as np
from pyrep import PyRep
from pyrep.objects.force_sensor import ForceSensor
from pyrep.robots.arms.panda import Panda
from pyrep.robots.end_effectors.panda_gripper import PandaGripper
from pyrep.objects.shape import Shape
from pyrep.backend import sim
from observation import Observation


class Env(gym.Env):
    def __init__(self, cnf):
        self.cnf = cnf.env
        self._launch()
        self._setup_robot()
        self._setup_shapes()
        self._set_objects_collidable()
        self._set_collections()
        self._setup_force_sensors()

        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(7, ))
        # TODO: need to be made more general for vision space
        obs = self._init_step()
        self.observation_space = gym.spaces.Box(low=-np.inf,
                                                high=np.inf,
                                                shape=obs.get().shape)

    def _init_step(self):
        """
        This method is needed bevause some observational values are only
        available after some target velocities have been set for the arm.
        Not calling a step bevore reset would cause somee entries in obs
        to be None.
        """
        self.step([0, 0, 0, 0, 0, 0, 0])
        obs = self.reset()
        return obs

    def _launch(self):
        scene_path = os.path.abspath(
            os.path.join('scenes', self.cnf.scene_path))
        self._pr = PyRep()
        self._pr.launch(scene_path, headless=self.cnf.headless)
        self._pr.start()

    def _setup_robot(self):
        self._arm = Panda()
        self._toggle_vel_control(True)
        self._gripper = PandaGripper()
        self._joint_start_positions = self._arm.get_joint_positions()
        self._gripper_start_positions = self._gripper.get_joint_positions()

    def _reset_robot(self):
        self._toggle_vel_control(False)
        self._arm.set_joint_positions(self._joint_start_positions)
        self._gripper.set_joint_positions(self._gripper_start_positions)
        self._toggle_vel_control(True)

    def _toggle_vel_control(self, is_velocity): 
        """
        Changes velocity control to *is_velocity*
        """
        self._arm.set_control_loop_enabled(not is_velocity)
        self._arm.set_motor_locked_at_zero_velocity(is_velocity)

    def _setup_shapes(self):
        self._table = Shape("diningTable_visible")
        self.concrete = Shape("Concrete")

    def _setup_force_sensors(self):
        self._fs0 = ForceSensor("force_sensor_0")
        self._fs1 = ForceSensor("force_sensor_1")
        self._fs2 = ForceSensor("force_sensor_2")
        self._fs3 = ForceSensor("force_sensor_3")
        self._fs4 = ForceSensor("force_sensor_4")

    def read_force_sensors(self):
        force_sensors = [self._fs0, self._fs1, self._fs2, self._fs3, self._fs4]
        return [sensor.read()[0] for sensor in force_sensors]

    def _set_vels(self, action):
        self._arm.set_joint_target_velocities(action)

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
        info = dict(collided=self.check_collision(), )
        return info

    def _get_observation(self):
        obs = Observation(
            self._arm.get_joint_velocities(), self._arm.get_joint_positions(),
            self._arm.get_joint_forces(), self._gripper.get_open_amount(),
            self._gripper.get_pose(), self._gripper.get_joint_positions(),
            self._gripper.get_touch_sensor_forces(),
            *self.read_force_sensors(),
            self._get_vision() if self.cnf.state == "vision" else None,
            self.cnf.state_size)
        return obs

    def _set_objects_collidable(self):
        self._arm.set_collidable(True)
        self._gripper.set_collidable(True)
        self._table.set_collidable(True)

    def _set_collections(self):
        self._collidables_collection = sim.simGetCollectionHandle(
            "collidables")
        self._robot_collection = sim.simGetCollectionHandle("Panda_arm")

    def step(self, action):
        self._set_vels(action)
        self._pr.step()
        obs = self._get_observation()
        reward = self._get_reward()
        done = self._get_done()
        info = self._get_info()
        return obs, reward, done, info

    def reset(self):
        """
        Not gym copliant reset function.
        """
        return self._get_observation()

    def render(self):
        # TODO: refer to RLBench gym implementation
        pass

    def close(self):
        self._pr.stop()
        self._pr.shutdown()

    def check_collision_with_table(self):
        return self._table.check_collision_by_handle(
            self._arm._collision_collection)

    def check_collision_with_concrete(self):
        return self.concrete.check_collision_by_handle(
            self._arm._collision_collection)

    def check_collision(self):
        """
        Checks whether the arm collides with the table or the slab.
        """
        other = sim.simCheckCollision(self._collidables_collection,
                                      self._robot_collection)
        # TODO: check for non trivial self collision
        # handle = sim.simGetCollisionHandle("Panda")
        # slf = sim.simReadCollision(handle)
        return other
