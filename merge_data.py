import pickle

datasets = []
for i in range(10):
    path = f"out/ds/iv_gen_dataset0_rank{i}_1dof.p"
    print("opening", path)
    with open(path, "rb") as f:
        datasets += pickle.load(f)

with open("out/dataset0_1dof.p", "wb") as f:
    pickle.dump(datasets, f)
