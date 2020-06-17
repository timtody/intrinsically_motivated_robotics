from .experiment import BaseExperiment
import altair as alt
import pandas as pd
import torch


class Experiment(BaseExperiment):
    """Implements Milestone 1 from inverse
    experiments catalogue.
    
    The purpose of this experiment is to show that after
    an agent has learned with mixin actions it can keep
    stable performance when mixing ratio is tuned to 0.

    For this use case, idealized actions are used i.e.
    given a goal the best action is computed by a hard coded
    scheme.
    """

    def __init__(self, cnf, rank):
        super().__init__(cnf, rank)
        self.results = []
        self.alpha_on = 30
        self.alpha_off = 120

    def compute_reward(self, goal, state):
        return -((goal - state) ** 2).sum()

    def run(self):
        self.env.reset()
        proto_action = [1] * self.cnf.env.action_dim

        # acquire goal
        for i in range(100):
            goal, *_ = self.env.step(proto_action)

        for i in range(self.cnf.main.n_steps):
            state = self.env.reset()
            done = False
            ep_len = 0

            if i == self.alpha_off:
                print("Changing alpha")
                self.agent.ppo.policy.alpha = 0
                self.agent.ppo.policy_old.alpha = 0

            if i == self.alpha_on:
                print("Changing alpha")
                self.agent.ppo.policy.alpha = 0.9
                self.agent.ppo.policy_old.alpha = 0.9

            while not done:
                ep_len += 1
                action = self.agent.get_action(
                    state, inverse_action=torch.tensor(proto_action)
                )
                state, *_ = self.env.step(action)
                distance = self.compute_reward(state, goal)
                reward = 0
                if distance > -5:
                    done = True
                    reward = 10

                if ep_len > 100:
                    done = True
                    reward = 0

                self.agent.set_reward(reward)
                self.agent.set_is_done(done)

            self.agent.train_ppo()
            self.results.append((self.rank, i, ep_len, self.alpha_on, self.alpha_off,))

        return self.results

    @staticmethod
    def plot(results):
        resuts_as_list = []
        for key, value in results.items():
            resuts_as_list += value
        df = pd.DataFrame(
            resuts_as_list, columns=["rank", "x", "ep_len", "alpha_on", "alpha_off"]
        )

        base = alt.Chart(df)
        chart = base.mark_line().encode(x="x:Q", y=alt.Y("ep_len", aggregate="mean",),)
        band = base.mark_errorband(extent="stdev").encode(
            alt.X("x:Q", title="Episode"), y=alt.Y("ep_len:Q", title="Episode length",),
        )
        line_on = base.mark_rule(color="green").encode(
            x=alt.X(("mean(alpha_on)"), axis=alt.Axis(title="Episode",),)
        )
        line_off = base.mark_rule(color="red").encode(
            x=alt.X(("mean(alpha_off)"), axis=alt.Axis(title="Episode",),)
        )

        (chart + band + line_on + line_off).show()
