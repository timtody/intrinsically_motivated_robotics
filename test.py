from env.environment import Env
from conf import get_conf


cnf = get_conf("conf/conf.yaml")
env = Env(cnf)
state = env.reset()
done = False
timestep = 0
while not done:
    timestep += 1
    env.step(env.action_space.sample())
    if timestep % 1000 == 0:
        env.close()
        env = Env(cnf)