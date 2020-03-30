from experiment import Behavior
from utils import get_conf

cnf = get_conf("conf/main.yaml")
exp = Behavior(cnf, 0)
exp.run()
