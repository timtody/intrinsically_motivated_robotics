from experiment import GoalReach
from utils import get_conf
from logging import Logger

cnf = get_conf("conf/main.yaml")
Logger(cnf)
exp = GoalReachAgent(cnf, 0)
exp.run()
