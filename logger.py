import os
import time
import json
from omegaconf import OmegaConf


class Logger:
    def __init__(self, cnf):
        self.cnf = cnf
        self._maybe_create_output_dir()
        path = self._get_path()
        self.owd = os.getcwd()
        os.environ["owd"] = self.owd
        cnf.log.owd = self.owd
        os.mkdir(path)
        os.chdir(path)
        self._make_exp_dirs()
        self._save_conf()

    def _save_conf(self):
        primitive = OmegaConf.to_container(self.cnf)
        with open("cnf/config.json", "w") as f:
            json.dump(primitive, f)

    def _get_path(self):
        if self.cnf.log.name:
            name_affix = f"-{self.cnf.log.name}"
        else:
            name_affix = ""
        path = os.path.join("out", time.strftime("%m-%d-%H-%M-%S") + name_affix)
        return path

    def _maybe_create_output_dir(self):
        if not os.path.exists("out"):
            print("Creating out dir")
            os.mkdir("out")

    def _make_exp_dirs(self):
        os.mkdir("data")
        os.mkdir("cnf")
        os.mkdir("checkpoints")
