from experiment import CheckActor
from mp import Runner
from utils import get_conf
from logger import Logger

cnf = get_conf("conf/main.yaml")
logger = Logger(cnf)
runner = Runner(CheckActor, cnf, "check_actor_ada_relu_big")
runner.run(5)
