import torch


def z_score1_normalization(feature, wn_size):
    """

    :param feature: size of nb_timeframe x nb_asset
    :param wn_size:
    :return: Output size: (nb_timeframe - wn_size) x nb_asset
    """
    # Shape: (nb_timeframe - no_wn_size + 1) x nb_asset x no_wn_size.
    data_unfold = feature.unfold(dimension=0, size=wn_size, step=1)

    mean = torch.mean(data_unfold, dim=2)  # Shape (nb_timeframe - wn_size + 1) x nb_asset.
    std = torch.std(data_unfold, dim=2)  # Sample deviation. Shape: the same with mean.

    # feature[wn_size:, :]: shape ((nb_timeframe - 1) - wn_size + 1) x nb_asset = (nb_timeframe - wn_size) x nb_asset.
    # First element: timeframe wn_size.
    # Last element: timeframe nb_timeframe -1.
    # mean[:-1, :] and std[-1, :]:
    # shape ((nb_timeframe - wn_size + 1) - 1) x nb_asset = (nb_timeframe - wn_size) x nb_asset.
    # First element: mean and std of timeframe from 0 to (wn_size - 1).
    # Last element: mean and std of timeframe from (nb_timeframe - wn_size - 1) to (nb_timeframe - 2)
    data_norm = (feature[wn_size:, :] - mean[:-1, :]) / std[:-1, :]  # Exclude the last element in mean and std.

    assert not torch.isinf(data_norm).any(), "There is inf value in normalization data. Increase normalized window."

    return data_norm


def z_score_block_normalization(feature, wn_size):
    """

    :param feature: nb_timeframe x nb_asset.
    :param wn_size:
    :return: output shape: (nb_timeframe - wn_size) x nb_asset.
    """
    data_norm = torch.zeros_like(feature)

    nb_timeframe = feature.shape[0]

    for tf in range(wn_size, nb_timeframe, wn_size):
        mean = torch.mean(feature[(tf - wn_size):tf], dim=0, keepdim=True)
        std = torch.std(feature[(tf - wn_size):tf], dim=0, keepdim=True)

        if tf + wn_size < nb_timeframe:
            # Final block of the data.
            data_norm[tf:(tf + wn_size), :] = (feature[tf:(tf + wn_size), :] - mean) / std
        else:
            data_norm[tf:] = (feature[tf:] - mean) / std

    data_norm = data_norm[wn_size:, ]

    return data_norm


def get_normalize_data(data, asset_return, no_wn_size, normalization_scheme):
    data_normalize = torch.zeros_like(data)
    return_feature = torch.zeros_like(asset_return)

    if normalization_scheme == "raw":
        data_normalize = data
        return_feature = asset_return
    elif normalization_scheme == "z_score1":
        data_normalize[(no_wn_size + 1):, :] = z_score1_normalization(data[1:, :], no_wn_size)
        return_feature[(no_wn_size + 1):, :] = z_score1_normalization(asset_return[1:, :], no_wn_size)
    else:
        data_normalize[(no_wn_size + 1):, :] = z_score_block_normalization(data[1:, :], no_wn_size)
        return_feature[(no_wn_size + 1):, :] = z_score_block_normalization(asset_return[1:, :], no_wn_size)

    return data_normalize, return_feature
