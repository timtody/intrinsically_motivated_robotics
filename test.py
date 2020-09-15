from environment import Env
from pyrep.objects.vision_sensor import VisionSensor
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
from pyrep.const import RenderMode


def get_conf(path):
    cnf = OmegaConf.load(path)
    cnf.merge_with_cli()
    OmegaConf.set_struct(cnf, True)
    return cnf


cnf = get_conf("conf/main.yaml")
env = Env(cnf)

env.reset()
print(env._arm.joints[0].get_joint_upper_velocity_limit())
exit(1)
for i in range(550):
    print(i)
    env.step([1, 0, 0, 0, 0, 0, 0])
    print(env._arm.get_joint_velocities())

