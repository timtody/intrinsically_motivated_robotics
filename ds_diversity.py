import torch
import matplotlib.pyplot as plt

dsim = torch.load("out/newdb4dof.p")
dsnoim = torch.load("out/newdb4dof-noim.p")


def measure_diversity(ds):
    results = []
    for state, nstate, _ in ds[100000:150000]:
        results.append((state - nstate).sum())

    return results


plt.plot(measure_diversity(dsim), label="im")
plt.plot(measure_diversity(dsnoim), label="noim")
plt.legend()
plt.show()
