from experiment import CheckActor
from mp import Runner
from utils import get_conf

cnf = get_conf("conf/main.yaml")
# exp = CheckActor(cnf=cnf, name="check_actor")
runner = Runner(CheckActor, cnf, "check_actor_2")
runner.run(10)