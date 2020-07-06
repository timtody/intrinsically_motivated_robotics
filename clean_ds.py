import pickle
import torch

for i in range(10):
    path = f"out/ds/iv_gen_dataset_prop_7dof_with-im_rank{i}.p"
    print("opening", path)
    with open(path, "rb") as f:
        file = pickle.load(f)
        sub_ds = []
        for (state, nstate, action) in file:
            sub_ds.append((state, nstate, action.tolist()))

    torch.save(sub_ds, f"out/ds/iv_gen_dataset_prop_7dof_with-im_rank{i}_cleaned.p")
