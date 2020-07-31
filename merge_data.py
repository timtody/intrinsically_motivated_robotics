import pickle
import torch

datasets = []
for i in range(0, 20):
    path = f"out/db/long/noreset-0newdb_3dof_with-im_rank{i}.p"
    print("opening", path)
    try:
        with open(path, "rb") as f:
            file = pickle.load(f)
            datasets += file
            del f
            del file
    except:
        print("path not found")

torch.save(datasets, "out/db-noreset-3dof-im.p")
