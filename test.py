from experiment import GoalReach
from utils import get_conf
from logger import Logger

cnf = get_conf("conf/main.yaml")
Logger(cnf)
exp = GoalReach(cnf, 0)
exp.run()
