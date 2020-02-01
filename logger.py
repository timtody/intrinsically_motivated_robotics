import os


class Logger:
    @staticmethod
    def setup(cnf):
        Logger._maybe_create_output_dir()
        path = Logger._get_path(cnf)
        os.mkdir(path)
        os.chdir(path)
        Logger._make_exp_dirs()

    @staticmethod
    def _get_path(cnf):
        path = os.path.join("output", cnf.log.name)
        return path

    @staticmethod
    def _maybe_create_output_dir():
        if not os.path.exists("output"):
            print("Creating output dir")
            os.mkdir("output")

    @staticmethod
    def _make_exp_dirs():
        os.mkdir("data")
        os.mkdir("vid")
