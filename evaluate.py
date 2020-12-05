import torch
import numpy as np


class IVEvaluator:
    def __init__(self):
        pass

    def evaluate(self, model: torch.nn.Module, tau: list, goals: list, reduce=False):
        if reduce:
            return self._compute_mean_error(model, tau, goals)
        else:
            return self._compute_error_across_time(model, tau, goals)

    def _compute_mean_error(self, model: torch.nn.Module, tau: list, goals: list):
        all_errors = []
        for t, goal in zip(tau, goals):
            total_error = 0
            for (state, nstate, action) in t:
                predicted_action = (
                    model(torch.tensor(state).float(), torch.tensor(goal).float())
                    .detach()
                    .numpy()
                )
                action = action / np.linalg.norm(action)
                predicted_action = predicted_action / np.linalg.norm(predicted_action)
                error = np.linalg.norm(predicted_action - action) ** 2
                total_error += error
            all_errors.append(total_error / len(tau))
        return np.array(all_errors)

    def _compute_error_across_time(
        self, model: torch.nn.Module, tau: list, goals: list
    ):
        all_errors = []

        for t, goal in zip(tau, goals):
            errors_in_t = []
            for (state, nstate, action) in t:
                predicted_action = (
                    model(
                        torch.tensor(state).float().to(model.device),
                        torch.tensor(goal).float().to(model.device),
                    )
                    .cpu()
                    .detach()
                    .numpy()
                )
                action = action / np.linalg.norm(action)
                predicted_action = predicted_action / np.linalg.norm(predicted_action)
                error = np.linalg.norm(predicted_action - action) ** 2
                errors_in_t.append(error)
            all_errors.append(errors_in_t)
        return np.array(all_errors)
