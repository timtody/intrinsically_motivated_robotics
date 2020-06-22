# grid_size = 5
# for ix, i in enumerate(range(1, grid_size + 1, 2)):
#     actions = []
#     print(i)
#     for j in range(i ** 2 - (i - 2) ** 2 if i > 1 else 1):
#         if ix == 0:
#             actions.append([0, 0])
#             actions.append("stop")
#             continue
#         if j // (ix + 1) == 0:
#             for i in range(ix):
#                 actions.append([0, 1])
#             for i in range(j % (ix + 1)):
#                 actions.append([1, 0])
#         elif j // (ix + 1) == 1:
#             for i in range(ix):
#                 actions.append([1, 0])
#             for i in range(j % (ix + 1)):
#                 actions.append([0, -1])
#         elif j // (ix + 1) == 2:
#             for i in range(ix):
#                 actions.append([0, -1])
#             for i in range(j % (ix + 1)):
#                 actions.append([-1, 0])
#         elif j // (ix + 1) == 3:
#             for i in range(ix):
#                 actions.append([-1, 0])
#             for i in range(j % (ix + 1)):
#                 actions.append([0, 1])
#         else:
#             print("bro")
#             print(j)

#         actions.append("stop")
#     print(actions)

import numpy as np


def compute_act_seq(side_len):
    """
    Computes the sequence of actions needed to check all
    positions around the agent in a square with side length
    side_len.
    """
    actions = []
    start_idx = side_len // 2
    for i in range(-start_idx, start_idx + 1):
        for j in range(-start_idx, start_idx + 1):
            actions += list(np.sign(i) * np.array(abs(i) * [[0, -1]]))
            actions += list(np.sign(j) * np.array(abs(j) * [[1, 0]]))
            actions += ["stop"]
    return actions


for act in compute_act_seq(5):
    print(act)
