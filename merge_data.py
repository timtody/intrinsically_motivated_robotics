import pickle

datasets = []
for i in range(10):
    path = f"out/iv_gen_dataset_rank{i}.pt"
    print("opening", path)
    with open(path, "rb") as f:
        datasets += pickle.load(f)

with open("out/dataset.p", "wb") as f:
    pickle.dump(datasets, f)
