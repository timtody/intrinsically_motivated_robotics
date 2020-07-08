import pickle
import torch

datasets = []
for i in range(50):
    path = f"out/db/iv_gen_dataset_prop_7dof_with-im_rank{i}.p"
    print("opening", path)
    with open(path, "rb") as f:
        file = pickle.load(f)
        datasets += file
        del f
        del file

torch.save(datasets, "out/full_ds_0_7dof_im.p")
