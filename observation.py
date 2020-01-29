import numpy as np


class Observation:
    # TODO: needs to be made more general (prob. dict type)
    # for when vision is introduced
    def __init__(
        self,
        joint_velocities,
        joint_positions,
        joint_forces,
        gripper_open_amount,
        gripper_pose,
        gripper_joint_positions,
        gripper_touch_forces,
        vision=None
    ):
        self.joint_velocities = joint_velocities
        self.joint_positions = joint_positions
        self.joint_forces = joint_forces
        self.gripper_open_amount = gripper_open_amount
        self.gripper_pose = gripper_pose
        self.gripper_joint_positions = gripper_joint_positions
        self.gripper_touch_forces = [tf for sublist in gripper_touch_forces
                                     for tf in sublist]
        self.rgb_left = self._maybe_get_vision(vision, "left")
        self.rgb_right = self._maybe_get_vision(vision, "right")
        self.rgb_wrist = self._maybe_get_vision(vision, "wrist")

    def _maybe_get_vision(self, vision, name):
        if vision is not None:
            return vision[name]
        return None

    def get_all(self):
        obs = []
        for key, data in self.__dict__.items():
            if data is not None:
                obs.append(data)
        return np.concatenate(obs)

    def get_stereo_vision(self):
        pass

    def get_eye_in_hand(self):
        pass
