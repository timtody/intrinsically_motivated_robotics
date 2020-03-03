from omegaconf import OmegaConf


def get_conf(path, struct=True, merge_cli=True):
    cnf = OmegaConf.load(path)
    if merge_cli:
        cnf.merge_with_cli()
    if struct:
        OmegaConf.set_struct(cnf, False)
    return cnf
