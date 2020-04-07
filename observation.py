import numpy as np
import torch

tac_mean = 0.001428180024959147
tac_std = 0.18175357580184937

prop_mean = -0.3062898814678192
prop_std = 26.55019760131836

audio_mean = -0.00512049812823534
audio_std = 0.4338988959789276

"""
tac:
	mean:0.0001090915611712262
	std:0.15071888267993927
prop:
	mean:-0.020761482417583466
	std:1.0171984434127808
audio:
	mean:0.1427653431892395
	std:1.0342168807983398

"""


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
        finger_left_forces,
        finger_right_forces,
        wrist_left_forces,
        wrist_right_forces,
        knuckle_forces,
        audio,
        vision=None,
        state_size="all",
    ):
        self.state_size = state_size

        self.joint_velocities = joint_velocities
        self.joint_positions = joint_positions
        # self.joint_forces = joint_forces
        # self.gripper_open_amount = gripper_open_amount
        # self.gripper_pose_prop = gripper_pose
        # self.gripper_joint_positions = gripper_joint_positions
        # self.gripper_touch_forces = [
        #     tf for sublist in gripper_touch_forces for tf in sublist
        # ]
        self.finger_left_forces_touch = finger_left_forces
        self.finger_right_forces_touch = finger_right_forces
        self.wrist_left_forces_touch = wrist_left_forces
        self.wrist_right_forces_touch = wrist_right_forces
        self.kuckle_forces_touch = knuckle_forces
        self.audio = np.array([*audio, *audio, *audio])
        self.rgb_left = self._maybe_extract_vision(vision, "left")
        self.rgb_right = self._maybe_extract_vision(vision, "right")
        self.rgb_wrist = self._maybe_extract_vision(vision, "wrist")

        # self._normalize()

    def _normalize(self):
        for key, data in self.__dict__.items():
            if data is not None and key != "state_size":
                data = np.array(data)
                if "touch" in key:
                    self.__setattr__(key, (data - tac_mean) / (tac_std + 1e-4))
                if "joint" in key:
                    self.__setattr__(key, (data - prop_mean) / (prop_std + 1e-4))
                if "audio" in key:
                    self.__setattr__(key, (data - audio_mean) / (audio_std + 1e-4))

    def _maybe_extract_vision(self, vision, name):
        if vision is not None:
            return vision[name]
        return None

    def get_all(self):
        obs = []
        for key, data in self.__dict__.items():
            if data is not None and key != "state_size":
                obs.append(data)
        return np.concatenate(obs)

    def get_filtered(self, filter):
        obs = []
        for key, data in self.__dict__.items():
            if data is not None and filter in key and key != "state_size":
                obs.append(data)
        return np.concatenate(obs)

    def get(self):
        if self.state_size == "all":
            return self.get_all()
        if self.state_size == "tac":
            return self.get_filtered("touch")
        if self.state_size == "prop":
            return self.get_filtered("joint")
        if self.state_size == "audio":
            return self.get_audio()

    def get_stereo_vision(self):
        pass

    def get_eye_in_hand(self):
        pass

    def __repr__(self):
        return str(self.get_all())

    def get_prop(self):
        return np.concatenate([v for k, v in self.__dict__.items() if "joint" in k])

    def get_tac(self):
        return np.concatenate([v for k, v in self.__dict__.items() if "touch" in k])

    def get_audio(self):
        return self.audio
