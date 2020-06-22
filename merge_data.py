import pickle

datasets = []
for i in range(10):
    path = f"out/ds/iv_gen_dataset_prop_rank{i}_2dof.p"
    print("opening", path)
    with open(path, "rb") as f:
        datasets += pickle.load(f)

with open("out/dataset0_2dof_prop.p", "wb") as f:
    pickle.dump(datasets, f)
