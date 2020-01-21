from rlbench.environment import Environment
from rlbench.action_modes import ActionMode, ArmActionMode
from rlbench.tasks import ReachTarget


class Env:
    def __init__(self, task=ReachTarget, action_mode="abs_vel"):
        action_mode = self._get_action_mode(action_mode)
        self.env = Environment(action_mode)
        self.task = self.env.get_task(task)

    def step(self, action):
        obs, reward, terminate = self.task.step(action)
        return obs, reward, terminate

    def render(self):
        pass

    def reset(self):
        descriptions, obs = self.task.reset()

    def _launch(self, headless=True):
        self.env.launch(headless=headless)

    def _get_action_mode(self, action_mode):
        if action_mode == "abs_vel":
            action_mode = ActionMode(ArmActionMode.ABS_JOINT_VELOCITY)
        return action_mode
