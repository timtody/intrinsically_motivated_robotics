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
viz = VisionSensor("viz")
viz_front = VisionSensor("viz_front")
env.reset()
env.step(env.action_space.sample())
# print(viz.get_resolution())
# print(viz.get_render_mode())
im = viz.capture_rgb()
plt.imshow(im)
plt.axis("off")
plt.savefig("side.pdf")
plt.show()

