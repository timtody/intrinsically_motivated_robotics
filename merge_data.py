import torch
import numpy as np

datasets = []
for i in range(50):
    path = f"out/ds/off-policy/ds_without_im_rank{i}.p"
    print("opening", path)
    ds = torch.load(path)
    datasets.append(ds)
    del ds

datasets = np.concatenate(datasets)

torch.save(datasets, "out/ds/off-policy/dataset_without_im")
