from experiment import CountCollisions
from utils import get_conf
from logger import Logger

cnf = get_conf("conf/main.yaml")
logger = Logger(cnf)
exp = CountCollisions(cnf, 0)
exp.run([])
