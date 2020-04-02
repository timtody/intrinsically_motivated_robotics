from pyrep import PyRep
from pyrep.objects.force_sensor import ForceSensor
from pyrep.robots.arms.panda import Panda
from pyrep.robots.end_effectors.panda_gripper import PandaGripper
from pyrep.objects.shape import Shape
from pyrep.backend import sim
from observation import Observation
import os
import gym
import math
import json
import numpy as np


class Env(gym.Env):
    def __init__(self, cnf):
        self.cnf = cnf.env
        self.owd = cnf.log.owd
        self._launch()
        self._setup_robot()
        self._setup_shapes()
        self._set_objects_collidable()
        self._set_collections()
        self._setup_force_sensors()

        self.action_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.cnf.action_dim,)
        )
        # TODO: need to be made more general for vision space
        obs = self._init_step()
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=obs.shape
        )
        # properties
        self.gripper_speed = 0
        self.sound_played = False

    def _init_step(self):
        """
        This method is needed bevause some observational values are only
        available after some target velocities have been set for the arm.
        Not calling a step bevore reset would cause somee entries in obs
        to be None.
        """
        self.step(self.cnf.action_dim * [0])
        obs = self.reset()
        return obs

    def _launch(self):
        scene_path = os.path.join(self.owd, "scenes", self.cnf.scene_path)
        self._pr = PyRep()
        self._pr.launch(os.path.abspath(scene_path), headless=self.cnf.headless)
        self._pr.start()

    def _setup_robot(self):
        self._arm = Panda()
        self._toggle_vel_control(True)
        self._gripper = PandaGripper()
        self._joint_start_positions = self._arm.get_joint_positions()
        self._gripper_start_positions = self._gripper.get_joint_positions()
        self._gripper_last_position = self.get_tip_position()

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
        self.head = Shape("Head")

    # def get_target_position(self):
    #     return self.target.get_position()

    def get_tip_position(self):
        return np.array(self._gripper.get_position())

    def _get_gripper_speed(self):
        current = self.get_tip_position()
        last = self._gripper_last_position
        self._gripper_last_position = current
        return np.linalg.norm(current - last)

    def _setup_force_sensors(self):
        self._fs0 = ForceSensor("force_sensor_0")
        self._fs1 = ForceSensor("force_sensor_1")
        self._fs2 = ForceSensor("force_sensor_2")
        self._fs3 = ForceSensor("force_sensor_3")
        self._fs4 = ForceSensor("force_sensor_4")

    def read_force_sensors(self):
        """
        Reads XYZ-forces from the force sensors attached to the robot
        arm. The gate function simply filters out values below a certain
        threshold to accomodate for motor noise.
        """
        return [
            list(map(self._gate, sensor.read()[0]))
            for sensor in [self._fs0, self._fs1, self._fs2, self._fs3, self._fs4]
        ]

    def get_sound_signal(self, threshold=0.025):
        # threshold = 0.035
        collided = self.check_collision()
        if not collided or self.gripper_speed < threshold:
            self.sound_played = False
            return np.array([0, 0, 0, 0])
        self.sound_played = True
        return self._compute_sound_signal2()

    def _compute_sound_signal(self):
        """
        Computes a sound vec of the form
        [loudness, distance, horizontal angle, vertical angle]
        where the angles are relative to the head position of the robot
        and the position where the collision creating the sound occured.
        """
        gripper_pos = self.get_tip_position()
        head_pos = np.array(self.head.get_position())
        # compute adjacent side
        adj = np.linalg.norm(gripper_pos[1] - head_pos[1])
        # compute opposite side
        opp = np.linalg.norm(gripper_pos[0] - head_pos[0])
        # compute the horizontal angle
        alpha = np.arctan(opp / adj)
        # compute adjacent side
        adj = opp
        # compute opposite side
        opp = np.linalg.norm(gripper_pos[2] - head_pos[2])
        # compute the vertical angle
        beta = np.arctan(opp / adj)
        # get the distance
        distance = np.linalg.norm(gripper_pos - head_pos)
        return [self.gripper_speed * 100, distance, alpha, beta]

    def _compute_sound_signal2(self):
        gripper_pos = self.get_tip_position()
        head_pos = np.array(self.head.get_position())
        radius = np.linalg.norm(gripper_pos - head_pos)
        theta = np.arccos(gripper_pos[2] / radius)
        if math.isnan(theta):
            theta = 0
        phi = np.arctan2(gripper_pos[1], gripper_pos[0])
        return np.array([self.gripper_speed * 100, radius, theta, phi])

    def _gate(self, x, threshold=0.01):
        if x < threshold:
            return 0
        return x

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
        info = dict(collided=self.check_collision(), sound=self.sound_played)
        return info

    def _get_observation(self):
        obs = Observation(
            self._arm.get_joint_velocities(),
            self._arm.get_joint_positions(),
            self._arm.get_joint_forces(),
            self._gripper.get_open_amount(),
            self._gripper.get_pose(),
            self._gripper.get_joint_positions(),
            self._gripper.get_touch_sensor_forces(),
            *self.read_force_sensors(),
            self.get_sound_signal(),
            self._get_vision() if self.cnf.state == "vision" else None,
            self.cnf.state_size
        )
        return obs.get()

    def _set_objects_collidable(self):
        self._arm.set_collidable(True)
        self._gripper.set_collidable(True)
        self._table.set_collidable(True)

    def _set_collections(self):
        self._collidables_collection = sim.simGetCollectionHandle("collidables")
        self._robot_collection = sim.simGetCollectionHandle("Panda_arm")

    def step(self, action):
        action = [*action, *((7 - self.cnf.action_dim) * [0])]
        self._set_vels(action)
        self.gripper_speed = self._get_gripper_speed()
        self._pr.step()
        return (
            self._get_observation(),
            self._get_reward(),
            self._get_done(),
            self._get_info(),
        )

    def reset(self, random=False):
        """
        Not gym copliant reset function.
        """
        if random:
            joint_start_pos = np.random.uniform(-1, 1, size=7)
            gripper_start_pos = np.random.uniform(-1, 1, size=2)
            self._gripper.set_joint_positions(gripper_start_pos)
            self._arm.set_joint_positions(joint_start_pos)
        else:
            self._gripper.set_joint_positions(self._gripper_start_positions)
            self._arm.set_joint_positions(self._joint_start_positions)
        return self._get_observation()

    def render(self):
        # TODO: refer to RLBench gym implementation
        pass

    def close(self):
        self._pr.stop()
        self._pr.shutdown()

    def check_collision_with_table(self):
        return self._table.check_collision_by_handle(self._arm._collision_collection)

    def check_collision_with_concrete(self):
        return self.concrete.check_collision_by_handle(self._arm._collision_collection)

    def check_collision(self):
        # TODO: check for non trivial self collision
        """
        Checks whether the arm collides with the table or the slab.
        """
        other = sim.simCheckCollision(
            self._collidables_collection, self._robot_collection
        )
        # handle = sim.simGetCollisionHandle("Panda")
        # slf = sim.simReadCollision(handle)
        return other

    def save_state(self, timestep):
        path = os.path.join("checkpoints", str(timestep))
        if not os.path.exists(path):
            os.mkdir(path)
        state = dict(
            joint_positions=self._arm.get_joint_positions(),
            gripper_positions=self._gripper.get_joint_positions(),
            joint_start_positions=self._joint_start_positions,
            gripper_start_position=self._gripper_start_positions,
            gripper_last_position=list(self._gripper_last_position),
            sound_played=self.sound_played,
            gripper_speed=self.gripper_speed,
            joint_target_velocities=self._arm.get_joint_target_velocities(),
            gripper_target_velocities=self._gripper.get_joint_target_velocities(),
        )
        with open(
            os.path.join("checkpoints", str(timestep), "env_state.json"), "w"
        ) as f:
            json.dump(state, f)

    def load_state(self, path):
        print("loading env state")
        abspath = os.path.abspath(os.environ["owd"])
        with open(os.path.join(abspath, path, "env_state.json"), "r") as f:
            state = json.load(f)
        self._toggle_vel_control(False)
        self._arm.set_joint_positions(state["joint_positions"])
        self._gripper.set_joint_positions(state["gripper_positions"])
        self._toggle_vel_control(True)
        self._joint_start_positions = state["joint_start_positions"]
        self._gripper_start_positions = state["gripper_start_position"]
        self._gripper_last_position = state["gripper_last_position"]
        self.sound_played = state["sound_played"]
        self.gripper_speed = state["gripper_speed"]
        self._arm.set_joint_target_velocities(state["joint_target_velocities"])
        self._gripper.set_joint_target_velocities(state["gripper_target_velocities"])
