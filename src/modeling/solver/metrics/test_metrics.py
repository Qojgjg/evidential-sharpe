import torch


def loglikelihood_dougbrion(evidence, target, **kwargs):
    alpha = evidence + 1
    S = torch.sum(alpha, dim=1, keepdim=True)
    loglikelihood_err = torch.sum((target - (alpha / S)) ** 2, dim=1, keepdim=True)
    loglikelihood_var = torch.sum(
        alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True
    )
    loglikelihood = loglikelihood_err + loglikelihood_var
    return loglikelihood.mean()


def kl_divergence_dougbrion(evidence, target, **kwargs):
    batch_size, nb_classes = evidence.shape

    alpha = target + (1 - target) * (evidence + 1)

    ones = torch.ones([1, nb_classes], dtype=torch.float32, device=torch.device("cuda"))
    sum_alpha = torch.sum(alpha, dim=1, keepdim=True)
    first_term = (
        torch.lgamma(sum_alpha)
        - torch.lgamma(alpha).sum(dim=1, keepdim=True)
        + torch.lgamma(ones).sum(dim=1, keepdim=True)
        - torch.lgamma(ones.sum(dim=1, keepdim=True))
    )
    second_term = (
        (alpha - ones)
        .mul(torch.digamma(alpha) - torch.digamma(sum_alpha))
        .sum(dim=1, keepdim=True)
    )
    kl = first_term + second_term
    return kl.mean()


