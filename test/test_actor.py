from experiment import GoalReach
from mp import Runner
from utils import get_conf
from logger import Logger

cnf = get_conf("conf/main.yaml")
logger = Logger(cnf)
exp = GoalReach(cnf, 0)
exp.run([])
# runner = Runner(GoalReach, cnf)
# runner.run(1)
