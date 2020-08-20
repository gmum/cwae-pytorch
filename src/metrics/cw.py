import numpy
import torch
import math as m


def cw(X: torch.Tensor, y: torch.Tensor = None):
    assert len(X.size()) == 2
    N, D = X.size()

    if y is None:
        y = __silverman_rule_of_thumb(N)

    K = 1.0/(2.0*D-3.0)

    A1 = __pairwise_distances(X)
    A = (1/(N**2)) * (1/torch.sqrt(y + K*A1)).sum()

    B1 = __euclidean_norm_squared(X, axis=1)
    B = (2/N)*((1/torch.sqrt(y + 0.5 + K*B1))).sum()

    return (1/m.sqrt(1+y)) + A - B


def cw_sampling(first_sample: torch.Tensor, second_sample: torch.Tensor, y: torch.Tensor = None):
    def phi_sampling(s, D):
        return (1.0 + 4.0*s/(2.0*D-3))**(-0.5)

    N_int, D_int = first_sample.size()
    N = float(N_int)
    D = float(D_int)

    if y is None:
        y = __silverman_rule_of_thumb(N_int)

    T = 1.0/(2.0*N*N*m.sqrt(m.pi*y))

    A0 = __pairwise_distances(first_sample)
    A = (phi_sampling(A0/(4*y), D)).sum()

    B0 = __pairwise_distances(second_sample)
    B = (phi_sampling(B0/(4*y), D)).sum()

    C0 = __pairwise_distances(first_sample, second_sample)
    C = (phi_sampling(C0/(4*y), D)).sum()

    return T*(A + B - 2*C)


def cw_choose(z_dim: int):
    if z_dim == 1:
        return cw_1d
    elif z_dim == 2:
        return cw_2d
    elif z_dim >= 8:
        return cw
    else:
        raise ValueError('Not available for this latent dimension')


def cw_1d(X: torch.Tensor, y: torch.Tensor = None):
    def N0(mean, variance):
        return 1.0/(torch.sqrt(2.0 * m.pi * variance)) * torch.exp((-(mean**2))/(2*variance))

    N = X.size(0)
    if y is None:
        y = __silverman_rule_of_thumb(N)

    A = X.unsqueeze(1) - X
    return (1.0/(N*N)) * N0(A, 2*y).sum() + N0(0.0, 2.0 + 2*y) - (2/N) * N0(X, 1.0 + 2*y).sum()


def cw_2d(X: torch.Tensor, y: torch.Tensor = None):
    def __phi(x):
        def __phi_f(s):
            t = s/7.5
            return torch.exp(-s/2) * (1 + 3.5156229*t**2 + 3.0899424*t**4 + 1.2067492*t**6 + 0.2659732*t**8
                                      + 0.0360768*t**10 + 0.0045813*t**12)

        def __phi_g(s):
            t = s/7.5
            return torch.sqrt(2/s) * (0.39894228 + 0.01328592*t**(-1) + 0.00225319*t**(-2) - 0.00157565*t**(-3)
                                      + 0.0091628*t**(-4) - 0.02057706*t**(-5) + 0.02635537*t**(-6) - 0.01647633*t**(-7)
                                      + 0.00392377*t**(-8))

        a = 7.5
        return __phi_f(torch.minimum(x, a)) - __phi_f(a) + __phi_g(torch.maximum(x, a))

    N = X.size(0)
    if y is None:
        y = __silverman_rule_of_thumb(N)

    A = 1/(N*N*torch.sqrt(y))
    B = 2.0/(N*torch.sqrt(y+0.5))

    A1 = __pairwise_distances(X)/(4*y)
    B1 = __euclidean_norm_squared(X, axis=1)/(2+4*y)
    return 1/torch.sqrt(1+y) + A*__phi(A1).sum() - B*__phi(B1).sum()


def __pairwise_distances(x: torch.Tensor, y: torch.Tensor = None) -> torch.Tensor:
    x_norm = (x**2).sum(1).view(-1, 1)
    if y is not None:
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y**2).sum(1).view(1, -1)
    else:
        y_t = torch.transpose(x, 0, 1)
        y_norm = x_norm.view(1, -1)

    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    return torch.clamp(dist, 0.0, numpy.inf)


def __silverman_rule_of_thumb(N: int):
    return (4/(3*N))**0.4


def __euclidean_norm_squared(X: torch.Tensor, axis: int = None):
    return (X**2).sum(axis)
