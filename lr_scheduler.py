from math import cos, pi, tanh
from functools import partial

import torch


__all__ = ["ConstantScheduler", "cycle_scheduler", "step_scheduler", "lr_finder"]


def anneal_linear(start, end, proportion):
    return start + proportion * (end - start)


def anneal_cos(start, end, proportion):
    cos_val = cos(pi * proportion) + 1

    return end + (start - end) / 2 * cos_val


def anneal_cospow(start, end, proportion):
    power = 5

    cos_val = 0.5 * (cos(pi * proportion) + 1) + 1
    cos_val = power ** cos_val - power
    cos_val = cos_val / (power ** 2 - power)

    return end + (start - end) * cos_val


def anneal_poly(start, end, proportion, power=0.9):
    return (start - end) * (1 - proportion) ** power + end


def anneal_tanh(start, end, proportion, lower=-6, upper=3):
    return end + (start - end) / 2 * (1 - tanh(lower + (upper - lower) * proportion))


def anneal_flat(start, end, proportion):
    return start


def anneal_exp(start, end, proportion):
    return start * (end / start) ** proportion


class ConstantScheduler:
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.lr = self.optimizer.param_groups[0]["lr"]

    def step(self):
        return self.lr


class PhaseScheduler:
    def __init__(self, optimizer, phases):
        self.optimizer = optimizer

        self.phase_param = phases

        self.lr_phase = self.make_phase(phases)

        self.phase = 0
        self.phase_step = 0

        self.latest_lr = None
        self.loss_log = []

    def make_phase(self, phases):
        phase_map = {
            "linear": anneal_linear,
            "cos": anneal_cos,
            "cospow": anneal_cospow,
            "poly": anneal_poly,
            "tanh": anneal_tanh,
            "exp": anneal_exp,
            "flat": anneal_flat,
        }

        lr_phase = []

        for phase in phases:
            if len(phase) == 4:
                phase_name, lr_from, lr_to, phase_iter = phase
                phase_fn = phase_map[phase_name]

            else:
                phase_name, lr_from, lr_to, phase_iter, phase_args = phase
                phase_fn = partial(phase_map[phase_name], **phase_args)

            lr_phase.append((lr_from, lr_to, phase_iter, phase_fn))

        return lr_phase

    def state_dict(self):
        return {
            "lr_phase": self.lr_phase,
            "phase_param": self.phase_param,
            "phase_step": self.phase_step,
            "latest_lr": self.latest_lr,
            "loss_log": self.loss_log,
        }

    def load_state_dict(self, state_dict):
        self.lr_phase = self.make_phase(state_dict["phase_param"])
        self.phase = state_dict["phase"]
        self.phase_step = state_dict["phase_step"]
        self.latest_lr = state_dict["latest_lr"]
        self.loss_log = state_dict["loss_log"]

    def __repr__(self):
        return f"PhaseScheduler(phases={self.lr_phase})"

    def step(self):
        if self.phase >= len(self.lr_phase):
            return

        # lr = self.lr_phase[self.phase].step()
        lr_from, lr_to, phase_iter, phase_fn = self.lr_phase[self.phase]
        self.phase_step += 1
        lr = phase_fn(lr_from, lr_to, self.phase_step / phase_iter)

        for group in self.optimizer.param_groups:
            group["lr"] = lr

        self.latest_lr = lr

        if self.phase_step > phase_iter:
            self.phase += 1
            self.phase_step = 0

        return lr

    def record_loss(self, loss):
        if isinstance(loss, torch.Tensor):
            loss = loss.item()

        self.loss_log.append((self.latest_lr, loss))

    def write_log(self, filename):
        with open(filename, "w") as f:
            for lr, loss in self.loss_log:
                f.write(f"{lr},{loss}\n")


def cycle_scheduler(
    optimizer,
    lr,
    n_iter,
    initial_multiplier=4e-2,
    final_multiplier=1e-5,
    warmup=500,
    plateau=0,
    decay=("cos", "cos"),
):
    phases = []

    if warmup > 0:
        phases.append((decay[0], lr * initial_multiplier, lr, warmup))

    if plateau > 0:
        phases.append(("linear", lr, lr, plateau))

    phases.append((decay[1], lr, lr * final_multiplier, n_iter - warmup - plateau))

    return PhaseScheduler(optimizer, phases)


def step_scheduler(
    optimizer, lr, milestones, gamma=0.1, warmup=0, warmup_multiplier=4e-2
):
    phases = []

    milestones = milestones.copy()

    steps = 0

    if warmup > 0:
        phases.append(("linear", lr * warmup_multiplier, lr, warmup))
        steps += warmup

    current_lr = lr

    for current, forward in zip(
        [steps] + milestones, milestones + [milestones[-1] + 1]
    ):
        phases.append(("linear", current_lr, current_lr, forward - current))

        current_lr *= gamma
        steps = current

    return PhaseScheduler(optimizer, phases)


def lr_finder(optimizer, lr_min, lr_max, n_iter, linear=False):
    decay = "linear" if linear else "exp"

    phases = [(decay, lr_min, lr_max, n_iter)]

    return PhaseScheduler(optimizer, phases)
