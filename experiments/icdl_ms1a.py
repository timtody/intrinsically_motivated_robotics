"""
Creation of dataset with 10M transitions.
Use both IM and no IM
"""
import torch
import matplotlib.pyplot as plt
from .experiment import BaseExperiment
from evaluate import IVEvaluator
import seaborn as sns
import matplotlib.style as style

style.use("ggplot")


# sns.set()
# sns.set_style("whitegrid")


class Experiment(BaseExperiment):
    def __init__(self, cnf, rank):
        super().__init__(cnf, rank)
        self.iv_evaluator = IVEvaluator()

    def generate_transition_with_goal(self, action, length=30):
        tau = []
        state = self.env.reset()

        for i in range(length):
            nstate, *_ = self.env.step(action)
            tau.append((state, nstate, action))
            state = nstate
        goal = state
        return tau, goal

    def run(self, pre_run_results):
        self.agent.icm.load_inverse_state(
            "out/ds/off-policy/iv-state-rank0relu-im-False"
        )
        taus = []
        goals = []

        # action = [0.3, 1, 0.2, 0.2, 0.3, 0.2, 0.2]
        # tau, goal = self.generate_transition_with_goal(action)
        # taus.append(tau)
        # goals.append(goal)

        # action = [-0.3, 1, 0.2, 0.2, 0.3, 0.2, 0.2]
        # tau, goal = self.generate_transition_with_goal(action)
        # taus.append(tau)
        # goals.append(goal)

        action = [1, 1, 0.2, 0.2, 0.3, 0.2, 0.2]
        tau, goal = self.generate_transition_with_goal(action, length=25)
        taus.append(tau)
        goals.append(goal)

        action = [-1, 1, 0.2, 0.2, 0.3, 0.2, 0.5]
        tau, goal = self.generate_transition_with_goal(action, length=25)
        taus.append(tau)
        goals.append(goal)

        action = [-0.8, 0.8, -1, -0.2, 0.3, 0.5, 0.2]
        tau, goal = self.generate_transition_with_goal(action, length=25)
        taus.append(tau)
        goals.append(goal)

        action = [-1, -1, 0.2, 0.2, 0.3, 0.2, 0.2]
        tau, goal = self.generate_transition_with_goal(action, length=25)
        taus.append(tau)
        goals.append(goal)

        action = [1, -1, 0.2, 0.2, 0.3, -0.5, 0.2]
        tau, goal = self.generate_transition_with_goal(action, length=25)
        taus.append(tau)
        goals.append(goal)

        action = [-1, 1, -0.2, 0.7, -0.3, -0.2, -0.2]
        tau, goal = self.generate_transition_with_goal(action, length=25)
        taus.append(tau)
        goals.append(goal)

        torch.save((taus, goals), "hard_goals.p")

        # tau = []
        # action = [-0.3, 0.8, -0.4, 0.1, -0.1, 0.2, 0.5]
        # state = self.env.reset()

        # for i in range(20):
        #     nstate, *_ = self.env.step(action)
        #     tau.append((state, nstate, action))
        #     state = nstate
        # taus.append(tau)
        # goals.append(nstate)

        # tau = []
        # action = [-0.1, 1, -0.3, 0.2, -0.1, 0.1, 0.1]
        # state = self.env.reset()

        # for i in range(20):
        #     nstate, *_ = self.env.step(action)
        #     tau.append((state, nstate, action))
        #     state = nstate
        # taus.append(tau)
        # goals.append(nstate)

        # print(self.iv_evaluator.evaluate(self.agent.icm._inverse, tau, state))
        results = self.iv_evaluator.evaluate(self.agent.icm._inverse, taus, goals)

        plt.bar(
            range(1, len(taus[0]) + 1),
            results.mean(axis=0),
            yerr=results.std(axis=0),
            color="#B1483B",
        )
        plt.ylabel("Error")
        plt.xlabel("Step")
        plt.show()

        # 12.162220628724185 3.2877194112877213
        # 0.9645769945048356 0.22412556069075174

