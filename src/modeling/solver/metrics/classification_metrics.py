import torch


def err(evidence: torch.Tensor, target: torch.Tensor, **kwargs):
    alpha = evidence + 1

    strength = alpha.sum(dim=-1)

    p = alpha / strength[:, None]

    err = (target - p) ** 2

    return err.sum(dim=-1).mean()


def var(evidence: torch.Tensor, **kwargs):
    alpha = evidence + 1

    strength = alpha.sum(dim=-1)

    p = alpha / strength[:, None]

    var = p * (1 - p) / (strength[:, None] + 1)

    return var.sum(dim=-1).mean()


def squared_error_bayes_risk(evidence: torch.Tensor, target: torch.Tensor, **kwargs):
    batch_size, nb_classes = evidence.shape

    alpha = evidence + 1

    strength = alpha.sum(dim=-1)

    p = alpha / strength[:, None]

    err = (target - p) ** 2
    assert err.shape == (batch_size, nb_classes), "Check err shape"

    var = p * (1 - p) / (strength[:, None] + 1)
    assert var.shape == (batch_size, nb_classes), "Check var shape"

    loss = (err + var).sum(dim=-1)

    return loss.mean()


def kl_divergence_loss(evidence: torch.Tensor, target: torch.Tensor, **kwargs):
    batch_size, nb_classes = evidence.shape

    alpha = evidence + 1

    alpha_tilde = target + (1 - target) * alpha

    strength_tilde = alpha_tilde.sum(dim=-1)

    first = torch.lgamma(alpha_tilde.sum(dim=-1)) \
            - torch.lgamma(torch.tensor(nb_classes)) \
            - (torch.lgamma(alpha_tilde)).sum(dim=-1)

    second = (
            (alpha_tilde - 1) *
            (torch.digamma(alpha_tilde) - torch.digamma(strength_tilde)[:, None])
    ).sum(dim=-1)

    loss = first + second

    return loss.mean()


def edl_loss(evidence, target, epoch_idx, annealing_step, **kwargs):
    annealing_coef = torch.min(
        torch.tensor(1.0, dtype=torch.float32),
        torch.tensor(epoch_idx / annealing_step, dtype=torch.float32)
    )

    bayes_risk = squared_error_bayes_risk(evidence, target)
    kl_div_loss = kl_divergence_loss(evidence, target)

    loss = bayes_risk + annealing_coef * kl_div_loss

    return loss


def uncertainty_per_samples(evidence: torch.Tensor, **kwargs):
    batch_size, nb_classes = evidence.shape
    alpha = evidence + 1

    u = nb_classes / torch.sum(alpha, dim=1, keepdim=True)
    assert u.shape == (batch_size, 1)

    return u


def match_vector(evidence: torch.Tensor, target: torch.Tensor, **kwargs):
    batch_size, nb_classes = evidence.shape

    pred = torch.argmax(evidence, dim=1)
    truth = torch.argmax(target, dim=1)

    match_vec = torch.reshape(torch.eq(pred, truth).float(), (-1, 1))
    assert match_vec.shape == (batch_size, 1)

    return match_vec


def mean_uncertainty_succ(evidence: torch.Tensor, target: torch.Tensor, **kwargs):
    match_vec = match_vector(evidence, target)
    u = uncertainty_per_samples(evidence)

    mean_u_succ = torch.sum(
        u * match_vec
    ) / torch.sum(match_vec + 1e-20)

    return mean_u_succ


def mean_uncertainty_fail(evidence: torch.Tensor, target: torch.Tensor, **kwargs):
    match_vec = match_vector(evidence, target)
    u = uncertainty_per_samples(evidence)

    mean_u_fail = torch.sum(
        u * (1 - match_vec)
    ) / (torch.sum(torch.abs(1 - match_vec)) + 1e-20)

    return mean_u_fail


def mean_evidence_succ(evidence: torch.Tensor, target: torch.Tensor, **kwargs):
    """
    Compute evidences across all classes when model make correct predictions.
    :param evidence:
    :param target:
    :return:
    """
    batch_size, nb_classes = evidence.shape
    match_vec = match_vector(evidence, target)

    mean_e_succ = torch.sum(
        torch.sum(evidence, 1, keepdim=True) * match_vec
    ) / torch.sum(match_vec + 1e-20)

    return mean_e_succ


def mean_evidence_fail(evidence: torch.Tensor, target: torch.Tensor, **kwargs):
    """

    :param evidence:
    :param target:
    :return:
    """
    batch_size, nb_classes = evidence.shape
    match_vec = match_vector(evidence, target)

    mean_e_fail = torch.sum(
        torch.sum(evidence, 1, keepdim=True) * (1 - match_vec)
    ) / (torch.sum(torch.abs(1 - match_vec)) + 1e-20)

    return mean_e_fail


def accuracy(evidence: torch.Tensor, target: torch.Tensor, **kwargs):
    batch_size, nb_classes = evidence.shape

    pred = torch.argmax(evidence, dim=1)
    assert pred.shape == (batch_size, ), "Check prediction label shape"

    truth = torch.argmax(target, dim=1)
    assert truth.shape == (batch_size, ), "Check ground truth label shape"

    acc = torch.sum(pred.eq(truth)) / truth.shape[0]

    return acc


def cross_entropy_loss(target: torch.Tensor, logit: torch.Tensor, **kwargs):
    labels = torch.argmax(target, dim=1)

    loss = torch.nn.CrossEntropyLoss()

    return loss(logit, labels)


