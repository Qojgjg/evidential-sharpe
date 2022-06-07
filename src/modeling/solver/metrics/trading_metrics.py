import torch
from torch.nn.functional import threshold


def portfolio_return(evidence,
                     future_return,
                     trans_rate,
                     device=torch.device("cuda"),
                     **kwargs):
    batch_size, nb_classes = evidence.shape
    pred = torch.argmax(evidence, dim=1, keepdim=True)

    port_return_vec = \
        (1 - trans_rate) * \
        (1 + torch.gather(future_return, dim=1, index=pred)) * \
        (1 - trans_rate) - 1
    assert port_return_vec.shape == (batch_size, 1), "Check portfolio return vector shape"
    return port_return_vec


def expected_returns(evidence,
                     future_return,
                     trans_rate,
                     device,
                     **kwargs):
    port_ret = portfolio_return(evidence, future_return, trans_rate, device)
    e_r = torch.mean(port_ret)

    return e_r


def standard_deviation(evidence,
                       future_return,
                       trans_rate,
                       device,
                       **kwargs):
    port_ret = portfolio_return(evidence, future_return, trans_rate, device)
    std = torch.std(port_ret)

    return std


def cal_sharpe_ratio(evidence,
                     future_return,
                     trans_rate,
                     device,
                     **kwargs):
    e_r = expected_returns(evidence, future_return, trans_rate, device)
    std = standard_deviation(evidence, future_return, trans_rate, device)
    sharpe = e_r / std

    return sharpe


def mean_cost(weight,
              future_return,
              trans_rate,
              device,
              **kwargs):
    mu = (1 - trans_rate) ** 2
    average_cost = torch.mean(1 - mu)

    return average_cost


def cumulative_return(weight,
                      future_return,
                      trans_rate,
                      log_scale,
                      device,
                      **kwargs):
    port_ret = portfolio_return(weight, future_return, trans_rate, device)
    cum_return = torch.cumprod(1 + port_ret, dim=0)

    if log_scale:
        cum_return = torch.log(cum_return)

    return cum_return


def fapv(weight,
         future_return,
         trans_rate,
         log_scale,
         device,
         **kwargs):
    cum_return = cumulative_return(weight, future_return, trans_rate, log_scale, device)

    return cum_return[-1]


def percentage_positive_return(weight,
                               future_return,
                               trans_rate,
                               device,
                               **kwargs):
    port_ret = portfolio_return(weight, future_return, trans_rate, device)
    pos_ret_num = torch.sum(port_ret > 0)
    pos_ret_ratio = pos_ret_num / port_ret.shape[0]

    return pos_ret_ratio


def pos_neg_return_ratio(weight,
                         future_return,
                         trans_rate,
                         device,
                         **kwargs):
    port_ret = portfolio_return(weight, future_return, trans_rate, device)

    avg_pos_ret = torch.mean(port_ret[port_ret > 0])
    avg_neg_ret = torch.mean(port_ret[port_ret <= 0])

    avg_ratio = avg_pos_ret / torch.abs(avg_neg_ret)

    return avg_ratio


def downside_deviation(weight,
                       future_return,
                       trans_rate,
                       mar,
                       device,
                       **kwargs):
    port_ret = portfolio_return(weight, future_return, trans_rate, device)
    semi_variance = torch.mean(torch.square(threshold(port_ret - mar, threshold=0, value=0)))
    semi_deviation = torch.sqrt(semi_variance)

    return semi_deviation


def sortino_ratio(weight,
                  future_return,
                  trans_rate,
                  mar,
                  device,
                  **kwargs):
    e_r = expected_returns(weight, future_return, trans_rate, device)
    dd = downside_deviation(weight, future_return, trans_rate, mar, device)

    sortino = (e_r - mar) / dd

    return sortino


def max_drawdown(weight,
                 future_return,
                 trans_rate,
                 device,
                 **kwargs):
    cum_ret = cumulative_return(weight, future_return, trans_rate, log_scale=False, device=device)
    previous_peak, _ = torch.cummax(cum_ret, dim=0)
    drawdowns = (cum_ret - previous_peak) / previous_peak
    max_dd = torch.min(drawdowns)

    return max_dd
