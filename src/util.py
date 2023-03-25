# Random functions / class definitions

import torch as th

device = th.device("cuda:0" if th.cuda.is_available() else "cpu")


def to_torch_tensor(t): return th.stack(t).to(device).detach()


def normalize(t, eps=1e-5):
    return (t - t.mean()) / (t.std() + eps)
