from pyrep import PyRep
from pyrep.objects.force_sensor import ForceSensor
from pyrep.robots.arms.panda import Panda
from pyrep.robots.end_effectors.panda_gripper import PandaGripper
from pyrep.objects.shape import Shape
from pyrep.objects.joint import Joint
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
        self._launch()
        self._setup_robot()
        self._setup_shapes()
        self._set_objects_collidable()
        self._set_collections()
        self._setup_force_sensors_hand()
        self._setup_mobile()
        self._setup_skin_fs()
        self._setup_skin_contacts()

        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(self.cnf.action_dim,))
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
        scene_path = os.path.join("scenes", self.cnf.scene_path)
        self._pr = PyRep()
        self._pr.launch(os.path.abspath(scene_path), headless=self.cnf.headless)
        self._pr.start()

    def _setup_robot(self):
        self._arm = Panda()
        self._set_vel_control(True)
        self._gripper = PandaGripper()
        self._joint_start_positions = self._arm.get_joint_positions()
        self._gripper_start_positions = self._gripper.get_joint_positions()
        self._gripper_last_position = self.get_tip_position()
        self._initial_robot_state = (
            self._arm.get_configuration_tree(),
            self._gripper.get_configuration_tree(),
        )

    def _reset_robot(self):
        self._set_vel_control(False)
        self._arm.set_joint_positions(self._joint_start_positions)
        self._gripper.set_joint_positions(self._gripper_start_positions)
        self._set_vel_control(True)

    def _set_vel_control(self, is_velocity):
        """
        Changes velocity control to *is_velocity*
        """
        self._arm.set_control_loop_enabled(not is_velocity)
        self._arm.set_motor_locked_at_zero_velocity(is_velocity)

    def _setup_shapes(self):
        self._table = Shape("diningTable_visible")
        self._concrete = Shape("Concrete")
        self._head = Shape("Head")

    def get_tip_position(self):
        return np.array(self._gripper.get_position())

    def _get_gripper_speed(self):
        current = self.get_tip_position()
        last = self._gripper_last_position
        self._gripper_last_position = current
        return np.linalg.norm(current - last)

    def _setup_force_sensors_hand(self):
        self._fs0 = ForceSensor("force_sensor_0")
        self._fs1 = ForceSensor("force_sensor_1")
        self._fs2 = ForceSensor("force_sensor_2")
        self._fs3 = ForceSensor("force_sensor_3")
        self._fs4 = ForceSensor("force_sensor_4")

        self.touch_sensors_hand = [
            self._fs0,
            self._fs1,
            self._fs2,
            self._fs3,
            self._fs4,
        ]

    def _enable_vel_control(self, joint):
        joint.set_control_loop_enabled(False)
        joint.set_motor_locked_at_zero_velocity(True)

    def _setup_mobile(self):

        """
        Sets up the joints of the robot mobile which consists of three
        spherical joints measuring rotation around XYZ-axis.
        """
        self._mobile_0_joint_0 = Joint("revolute_beta")
        self._mobile_0_joint_1 = Joint("revolute_gamma")
        self._mobile_1_joint_0 = Joint("revolute_beta#0")
        self._mobile_1_joint_1 = Joint("revolute_gamma#0")
        self._mobile_2_joint_0 = Joint("revolute_beta#1")
        self._mobile_2_joint_1 = Joint("revolute_gamma#1")

        self._mobile_collection = [
            self._mobile_0_joint_0,
            self._mobile_0_joint_1,
            self._mobile_1_joint_0,
            self._mobile_1_joint_1,
            self._mobile_2_joint_0,
            self._mobile_2_joint_1,
        ]

    def _setup_skin_contacts(self):
        """
        Sets up the contacts which connect to the force sensors. These are
        the shapes which collide and interact with other objects in the scene.
        """
        self._skin_contact_base_0 = Shape("right_wrist_force_contact#8")
        self._skin_contact_base_1 = Shape("right_wrist_force_contact#9")
        self._skin_contact_base_2 = Shape("right_wrist_force_contact#10")
        self._skin_contact_base_3 = Shape("right_wrist_force_contact#18")
        self.skin_contacts_base = [
            self._skin_contact_base_0,
            self._skin_contact_base_1,
            self._skin_contact_base_2,
            self._skin_contact_base_3,
        ]

        self._skin_contact_upper_0 = Shape("right_wrist_force_contact#0")
        self._skin_contact_upper_1 = Shape("right_wrist_force_contact#1")
        self._skin_contact_upper_2 = Shape("right_wrist_force_contact#2")
        self._skin_contact_upper_3 = Shape("right_wrist_force_contact#3")
        self._skin_contact_upper_4 = Shape("right_wrist_force_contact#4")
        self._skin_contact_upper_5 = Shape("right_wrist_force_contact#5")
        self._skin_contact_upper_6 = Shape("right_wrist_force_contact#6")
        self._skin_contact_upper_7 = Shape("right_wrist_force_contact#7")
        self.skin_contact_upper = [
            self._skin_contact_upper_0,
            self._skin_contact_upper_1,
            self._skin_contact_upper_2,
            self._skin_contact_upper_3,
            self._skin_contact_upper_4,
            self._skin_contact_upper_5,
            self._skin_contact_upper_6,
            self._skin_contact_upper_7,
        ]

        self._skin_contact_forearm_0 = Shape("right_wrist_force_contact#11")
        self._skin_contact_forearm_1 = Shape("right_wrist_force_contact#12")
        self._skin_contact_forearm_2 = Shape("right_wrist_force_contact#13")
        self._skin_contact_forearm_3 = Shape("right_wrist_force_contact#14")
        self._skin_contact_forearm_4 = Shape("right_wrist_force_contact#15")
        self._skin_contact_forearm_5 = Shape("right_wrist_force_contact#16")
        self._skin_contact_forearm_6 = Shape("right_wrist_force_contact#17")
        self.skin_contact_forearm = [
            self._skin_contact_forearm_0,
            self._skin_contact_forearm_1,
            self._skin_contact_forearm_2,
            self._skin_contact_forearm_3,
            self._skin_contact_forearm_4,
            self._skin_contact_forearm_5,
            self._skin_contact_forearm_6,
        ]

        self._skin_contact_wrist_0 = Shape("right_wrist_force_contact#19")
        self._skin_contact_wrist_1 = Shape("right_wrist_force_contact#23")
        self._skin_contact_wrist_2 = Shape("right_wrist_force_contact#24")
        self._skin_contact_wrist_3 = Shape("right_wrist_force_contact#25")
        self.skin_contact_wrist = [
            self._skin_contact_wrist_0,
            self._skin_contact_wrist_1,
            self._skin_contact_wrist_2,
            self._skin_contact_wrist_3,
        ]

        self.skin_contact_hand_0 = Shape("right_outer_force_contact")
        self.skin_contact_hand_1 = Shape("left_outer_force_contact")
        self.skin_contact_hand_2 = Shape("palm_force_contact")
        self.skin_contact_hand_3 = Shape("left_wrist_force_contact")
        self.skin_contact_hand_4 = Shape("right_wrist_force_contact")
        self.skin_contact_hand = [
            self.skin_contact_hand_0,
            self.skin_contact_hand_1,
            self.skin_contact_hand_2,
            self.skin_contact_hand_3,
            self.skin_contact_hand_4,
        ]

        self.skin_contacts = [
            *self.skin_contacts_base,
            *self.skin_contact_upper,
            *self.skin_contact_forearm,
            *self.skin_contact_wrist,
            *self.skin_contact_hand,
        ]

    def _setup_skin_fs(self):
        """
        Sets up the skin sensors of the robot which are force sensors
        measuring forces in XYZ-direction. The skin is separated into three
        different groups (base, upper arm, forearm) to investigate
        how the preference of the agent for different parts may differ.
        This is also the place where we would hook in the idea that different
        parts may have different sensory density and will be preferred / neglected.
        """

        # base of arm
        self._skin_sensor_base_0 = ForceSensor("force_sensor_2#8")
        self._skin_sensor_base_1 = ForceSensor("force_sensor_2#9")
        self._skin_sensor_base_2 = ForceSensor("force_sensor_2#10")
        self._skin_sensor_base_3 = ForceSensor("force_sensor_2#18")
        self.skin_base = [
            self._skin_sensor_base_0,
            self._skin_sensor_base_1,
            self._skin_sensor_base_2,
            self._skin_sensor_base_3,
        ]

        # upper arm
        self._skin_sensor_upper_0 = ForceSensor("force_sensor_2#0")
        self._skin_sensor_upper_1 = ForceSensor("force_sensor_2#1")
        self._skin_sensor_upper_2 = ForceSensor("force_sensor_2#2")
        self._skin_sensor_upper_3 = ForceSensor("force_sensor_2#3")
        self._skin_sensor_upper_4 = ForceSensor("force_sensor_2#4")
        self._skin_sensor_upper_5 = ForceSensor("force_sensor_2#5")
        self._skin_sensor_upper_6 = ForceSensor("force_sensor_2#6")
        self._skin_sensor_upper_7 = ForceSensor("force_sensor_2#7")
        self.skin_upper = [
            self._skin_sensor_upper_0,
            self._skin_sensor_upper_1,
            self._skin_sensor_upper_2,
            self._skin_sensor_upper_3,
            self._skin_sensor_upper_4,
            self._skin_sensor_upper_5,
            self._skin_sensor_upper_6,
            self._skin_sensor_upper_7,
        ]

        # forearm
        self._skin_sensor_forearm_0 = ForceSensor("force_sensor_2#12")
        self._skin_sensor_forearm_1 = ForceSensor("force_sensor_2#13")
        self._skin_sensor_forearm_2 = ForceSensor("force_sensor_2#14")
        self._skin_sensor_forearm_3 = ForceSensor("force_sensor_2#15")
        self._skin_sensor_forearm_4 = ForceSensor("force_sensor_2#16")
        self._skin_sensor_forearm_5 = ForceSensor("force_sensor_2#17")
        self._skin_sensor_forearm_6 = ForceSensor("force_sensor_2#11")
        self.skin_forearm = [
            self._skin_sensor_forearm_0,
            self._skin_sensor_forearm_1,
            self._skin_sensor_forearm_2,
            self._skin_sensor_forearm_3,
            self._skin_sensor_forearm_4,
            self._skin_sensor_forearm_5,
            self._skin_sensor_forearm_6,
        ]

        # wrist
        self._skin_sensor_wrist_0 = ForceSensor("force_sensor_2#19")
        self._skin_sensor_wrist_1 = ForceSensor("force_sensor_2#23")
        self._skin_sensor_wrist_2 = ForceSensor("force_sensor_2#24")
        self._skin_sensor_wrist_3 = ForceSensor("force_sensor_2#25")
        self.skin_wrist = [
            self._skin_sensor_wrist_0,
            self._skin_sensor_wrist_1,
            self._skin_sensor_wrist_2,
            self._skin_sensor_wrist_3,
        ]

        self.skin = [
            *self.skin_base,
            *self.skin_upper,
            *self.skin_forearm,
            *self.skin_wrist,
        ]

    def get_joint_intervals(self):
        limits = []
        for joint in self._arm.joints:
            limits.append(joint.get_joint_interval()[1])
        return limits

    def get_joint_angles(self):
        return self._arm.get_joint_positions()

    def get_skin_info(self):
        """
        Retrieves the XYZ-forces of the skin sensors. Since the read method
        returns a tuple of forces and torques, we index into the first element.
        """

        # nested list comprehension flattens list
        return [e for skin_sensor in self.skin for e in skin_sensor.read()[0]]

    def get_skin_touch_map(self):
        """
        Returns a binary map representing all skin contacts. Each entry is 1,
        when it is in contact with something, else 0.
        """
        arm_contacts = [
            *self.skin_contact_wrist,
            *self.skin_contact_forearm,
            *self.skin_contact_upper,
            *self.skin_contacts_base,
        ]
        arm_map = [
            self._check_collision(contact.get_handle(), self._robot_collection_hand)
            for contact in arm_contacts
        ]
        hand_map = [
            self._check_collision(contact.get_handle(), self._all_but_hand_collection)
            for contact in self.skin_contact_hand
        ]
        return [*arm_map, *hand_map]

    def get_mobile_positions(self):
        return (
            self._mobile_0_joint_0.get_joint_position(),
            self._mobile_0_joint_1.get_joint_position(),
            self._mobile_1_joint_0.get_joint_position(),
            self._mobile_1_joint_1.get_joint_position(),
            self._mobile_2_joint_0.get_joint_position(),
            self._mobile_2_joint_1.get_joint_position(),
        )

    def get_mobile_velocities(self):
        return (
            self._mobile_0_joint_0.get_joint_velocity(),
            self._mobile_0_joint_1.get_joint_velocity(),
            self._mobile_1_joint_0.get_joint_velocity(),
            self._mobile_1_joint_1.get_joint_velocity(),
            self._mobile_2_joint_0.get_joint_velocity(),
            self._mobile_2_joint_1.get_joint_velocity(),
        )

    def get_mobile_forces(self):
        return (
            self._mobile_0_joint_0.get_joint_force(),
            self._mobile_0_joint_1.get_joint_force(),
            self._mobile_1_joint_0.get_joint_force(),
            self._mobile_1_joint_1.get_joint_force(),
            self._mobile_2_joint_0.get_joint_force(),
            self._mobile_2_joint_1.get_joint_force(),
        )

    def read_force_sensors_hand(self):
        """
        Reads XYZ-forces from the force sensors attached to the robot
        arm. The gate function simply filters out values below a certain
        threshold to accomodate for motor noise.
        """
        return [e for sensor in self.touch_sensors_hand for e in sensor.read()[0]]

    def get_sound_signal(self, threshold=0.025):
        # threshold = 0.035
        collided = self.check_collision()
        if not collided or self.gripper_speed < threshold:
            self.sound_played = False
            return np.array([0, 0, 0, 0])
        self.sound_played = True
        return self._compute_sound_signal()

    def _compute_sound_signal(self):
        gripper_pos = self.get_tip_position()
        head_pos = np.array(self._head.get_position())
        radius = np.linalg.norm(gripper_pos - head_pos)
        theta = np.arccos(gripper_pos[2] / radius)
        if math.isnan(theta):
            theta = 0
        phi = np.arctan2(gripper_pos[1], gripper_pos[0])
        return np.array([self.gripper_speed * 100, radius, theta, phi])

    def _gate(self, x, threshold=0.0):
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
        info = dict(
            collided_other=self.check_collision(),
            collided_self=self.check_collision_with_self(),
            collided_dyn=self.check_collision_with_dynamic(),
            # TODO: find real solution for handling get_info status
            # sound=self.sound_played,
        )
        return info

    def _get_observation(self):
        obs = []
        if "prop" in self.cnf.state:
            [obs.append(vel) for vel in self._arm.get_joint_velocities()]
            [obs.append(pos) for pos in self._arm.get_joint_positions()]
            [obs.append(pos) for pos in self._gripper.get_joint_positions()]
            [obs.append(open_amount) for open_amount in self._gripper.get_open_amount()]

        if "tac" in self.cnf.state:
            # switch = lambda x: 1 if x > 0.01 else 0
            # [obs.append(switch(reading)) for reading in self.read_force_sensors_hand()]
            # [obs.append(switch(skin_reading)) for skin_reading in self.get_skin_info()]
            [obs.append(touch) for touch in self.get_skin_touch_map()]

        if "mobile" in self.cnf.state:
            [obs.append(mob_vel) for mob_vel in self.get_mobile_velocities()]

        if "audio" in self.cnf.state:
            # TODO: implement this
            pass

        if "notrain" in self.cnf.state:
            # this is done to prevent pytorch from dividing by 0 in PPO
            # which will be the case because PPO accesses state.length
            # which would be 0 elsewise. During the notrain regime this
            # state is not used anyways so we can append stuff here.
            obs.append(1)

        # return (np.array(obs) - 0.1279311) / 4.236529262154189
        # TODO: REMOVE THE NORAMLIZATION OF THE OBSERVATION AFTER
        # EVALUATING FWMODEL
        # return np.array(obs)
        return np.array(obs) * 10
        # ---------------------------

    def _set_objects_collidable(self):
        self._arm.set_collidable(True)
        self._gripper.set_collidable(True)
        self._table.set_collidable(True)

    def _set_collections(self):
        self._collidables_collection = sim.simGetCollectionHandle("collidables")
        self._robot_collection = sim.simGetCollectionHandle("Panda_arm")
        self._robot_collection_hand = sim.simGetCollectionHandle("collection_hand")
        self._robot_collection_rest = sim.simGetCollectionHandle("collection_rest")
        self._collidables_dynamic_collection = sim.simGetCollectionHandle(
            "collection_dynamic"
        )
        self._all_but_hand_collection = sim.simGetCollectionHandle(
            "collection_all_but_hand"
        )

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
        """Resets the joint angles. """
        self._gripper.release()
        arm, gripper = self._initial_robot_state

        self._pr.set_configuration_tree(arm)
        self._pr.set_configuration_tree(gripper)
        self._arm.set_joint_positions(self._joint_start_positions)
        self._arm.set_joint_target_velocities([0] * len(self._arm.joints))
        self._gripper.set_joint_positions(self._gripper_start_positions)
        self._gripper.set_joint_target_velocities([0] * len(self._gripper.joints))

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
        return self._concrete.check_collision_by_handle(self._arm._collision_collection)

    def check_collision_with_self(self):
        """
        Checks if robot collides with itself. This is done by comparing
        the collection "hand" with the collection "rest".
        
        Needed, because checking collision with itself always yields true
        when acting on the whole robot collection.
        """

        return sim.simCheckCollision(
            self._robot_collection_hand, self._robot_collection_rest
        )

    def check_collision_with_other(self):
        """
        Check if the robot collided with something other than itself.
        Fits better with the notation of the check_collision_with_self method.
        """

        return sim.simCheckCollision(
            self._collidables_collection, self._robot_collection
        )

    def check_collision_with_dynamic(self):
        """
        Check if the robot collides with dynamic objects in the scence. So far
        these dynamic objects are the pendulums but could be other interactable
        objects like balls etc.
        """

        return sim.simCheckCollision(
            self._collidables_dynamic_collection, self._robot_collection
        )

    def _check_collision(self, enitity1, entity2):
        """
        generall checks for collision between two objects
        """
        return sim.simCheckCollision(enitity1, entity2)

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
        with open(os.path.join(path, "env_state.json"), "r") as f:
            state = json.load(f)
        self._set_vel_control(False)
        self._arm.set_joint_positions(state["joint_positions"])
        self._gripper.set_joint_positions(state["gripper_positions"])
        self._set_vel_control(True)
        self._joint_start_positions = state["joint_start_positions"]
        self._gripper_start_positions = state["gripper_start_position"]
        self._gripper_last_position = state["gripper_last_position"]
        self.sound_played = state["sound_played"]
        self.gripper_speed = state["gripper_speed"]
        self._arm.set_joint_target_velocities(state["joint_target_velocities"])
        self._gripper.set_joint_target_velocities(state["gripper_target_velocities"])
