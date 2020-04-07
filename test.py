from experiment import GoalReachAgent
from utils import get_conf
from logger import Logger

cnf = get_conf("conf/main.yaml")
Logger(cnf)
exp = GoalReachAgent(cnf, 0)
exp.run()
