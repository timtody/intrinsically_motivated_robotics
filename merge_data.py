import pickle
import torch

datasets = []
for i in range(7):
    path = f"out/ds/iv_gen_dataset_prop_7dof_with-im_rank{i}.p"
    print("opening", path)
    with open(path, "rb") as f:
        file = pickle.load(f)
        datasets += file
        del f
        del file

torch.save(datasets, "out/dataset0_7dof_im.p")
