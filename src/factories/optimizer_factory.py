from torch.optim import Adam


def get_optimizer_factory(parameters: dict, lr: float):
    return Adam(parameters, lr=lr)
