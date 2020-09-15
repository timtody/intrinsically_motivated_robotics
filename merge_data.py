import pickle
import torch

datasets = []
for i in range(1, 20, 2):
    path = f"out/db/long/new-5dof{i}.p"
    print("opening", path)
    try:
        with open(path, "rb") as f:
            file = pickle.load(f)
            datasets += file
            del f
            del file
    except:
        print("path not found")

torch.save(datasets, "out/5dof-noim.p")
