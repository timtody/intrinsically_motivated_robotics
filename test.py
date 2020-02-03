from env.environment import Env
from conf import get_conf
from utils import ReturnWindow

cnf = get_conf("conf/cnt_col.yaml")
env = Env(cnf)
state = env.reset()
done = False
timestep = 0
win = ReturnWindow(lookback=50)

# for i in range(32):
#     obs, *_, info = env.step([0, 0, 0, 0, 0, 1, -1])

for i in range(10):
    env.step([0, 0, 0, 0, 0, -1, 0])

while not done:
    timestep += 1
    obs, *_, info = env.step([0, 1, 0, 0, 0, 0, 0])
    relevant_forces = obs.get()[-15:]
    win.update(relevant_forces)
