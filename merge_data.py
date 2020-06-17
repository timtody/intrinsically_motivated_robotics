import pickle

datasets = []
for i in range(3):
    path = f"out/iv_gen_dataset_rank{i}_2dof.p"
    print("opening", path)
    with open(path, "rb") as f:
        datasets += pickle.load(f)

with open("out/dataset_2dof.p", "wb") as f:
    pickle.dump(datasets, f)
