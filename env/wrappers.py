import gym
from pyrep.objects.vision_sensor import VisionSensor
from pyrep.objects.dummy import Dummy
from pyrep.const import RenderMode


class ObsWrapper(gym.Wrapper):
    def __init__(self, env):
        super(ObsWrapper, self).__init__(env)
        self.render_cam = None

    def step(self, action):
        obs, reward, terminate = self.task.step(action)
        return obs, reward, terminate, None

    def reset(self):
        descriptions, obs = self.task.reset()
        del descriptions  # Not used.
        return obs

    def render(self, mode="human"):
        # self.env.render()
        if self.render_cam is None:
            # Add the camera to the scene
            cam_placeholder = Dummy('cam_cinematic_placeholder')
            self.render_cam = VisionSensor.create([256, 144])
            self.render_cam.set_pose(cam_placeholder.get_pose())
            self.render_cam.set_render_mode(RenderMode.OPENGL)

        if mode == "rgb_array":
            return self.render_cam.capture_rgb()
