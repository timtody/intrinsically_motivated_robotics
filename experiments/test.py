from .experiment import BaseExperiment


class Experiment(BaseExperiment):
    def run(self):
        obs = self.env.reset()
        print(len(obs))
        exit(1)
        for i in range(15):
            self.env.step([0, 0, 0, 0, 0, 1, 0])

        for i in range(200):
            self.env.step([0, 1, 0, 0, 0, 0, 0])

        self.env.close()
