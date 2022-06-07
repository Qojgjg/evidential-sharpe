import pandas as pd
import torch

from src.data.preprocessing.normalization import get_normalize_data

import torch
import torch.nn.functional as F


def generate_label(future_return):
    positions = torch.argmax(future_return, dim=1)
    labels = F.one_hot(positions, num_classes=2)
    return labels


def load_data(data_dir, index_range):
    """

    :param data_dir: path of raw data in .csv file format.
    :param index_range: list of two integer specified the range of sample
    used for a dataset.
    :return:
    """
    start_index = index_range[0]
    end_index = index_range[1]

    df = pd.read_csv(data_dir)

    # Remove timestamp column.
    if "timestamp" in df:
        df = df.drop(columns="timestamp")

    data = torch.tensor(df.values)

    return data[start_index:end_index, :]


def returns_calculation(closed_price):
    """
    Calculate returns of each asset using close price.
    :param closed_price: Shape nb_sample x nb_asset.
    :return: returns: Shape (nb_sample - 1) x nb_asset.
    """
    returns = (closed_price[1:, :] / closed_price[0:-1, :]) - 1
    return returns


def normalize_and_return_calculation(data,
                                     normalization_scheme,
                                     no_wn_size,
                                     lb_wn_size,
                                     ohlcv_feature):
    data_len = data.shape[0]
    nb_asset = data.shape[1] // 5

    # NOTE: EXCEPT FOR self.training_input AND self.future_return, OTHER INTERMEDIATE TENSORS HAS THE SAME LENGTH
    # IN TIME DIMENSION WITH THE ORIGINAL DATA.

    # Time index 0 will not be used since it does not have corresponding return feature.
    # Time index from 1 to no_wn_size must not be used later since it cannot be normalized.
    # For the raw normalize scheme, we still skip index 1 to no_wn_size to keep the code easy to understand.
    asset_return = torch.zeros((data_len, nb_asset))
    asset_return[1:, :] = returns_calculation(data[:, 2::5])

    # Normalize the raw ohlcv and returns data by given normalization method.
    data_normalize, return_feature = get_normalize_data(data=data,
                                                        asset_return=asset_return,
                                                        no_wn_size=no_wn_size,
                                                        normalization_scheme=normalization_scheme)

    nb_ohlcv_feature = int(sum(ohlcv_feature.values()))
    # Adding 1 accounts for return feature.
    data_selected_feature = torch.zeros((data_len, nb_asset * (nb_ohlcv_feature + 1)))
    data_selected_feature[:, nb_ohlcv_feature::(nb_ohlcv_feature + 1)] = return_feature

    for ind_asset in range(nb_asset):
        i = 0
        while i < nb_ohlcv_feature:
            for feat_ind, feat_name in enumerate(ohlcv_feature):
                if ohlcv_feature[feat_name]:
                    # 5 is the total number of feature in ohlcv format.
                    data_selected_feature[:, ind_asset * (nb_ohlcv_feature + 1) + i] = \
                        data_normalize[:, ind_asset * 5 + feat_ind]
                    i += 1

    # Shape: (data_len - no_wn_size - lb_wn_size - 1) x nb_asset.
    # Adding 1 in the index accounts for this is the FUTURE return.
    future_return = asset_return[(no_wn_size + lb_wn_size + 1):, :]

    return data_selected_feature, future_return


def make_window(data,
                data_selected_feature,
                no_wn_size,
                lb_wn_size,
                dimension_order,
                add_channel_dim):

    data_len = data.shape[0]

    nb_window = data_len - no_wn_size - lb_wn_size - 1
    # Reference to the original data time index.
    # First window: no_wn_size + 1 to no_wn_size + lb_wn_size
    # Last window: data_len - lb_wn_size to data_len - 1.
    training_input = torch.zeros((nb_window, lb_wn_size, data_selected_feature.shape[1]))

    window_index = [(start, start + lb_wn_size) for start in range(no_wn_size + 1, data_len - lb_wn_size)]
    for j in range(nb_window):
        start_ind, end_ind = window_index[j]
        training_input[j, :] = data_selected_feature[start_ind:end_ind, :]

    if add_channel_dim:
        training_input = torch.unsqueeze(training_input, dim=1)

    if dimension_order == 'spatial-temporal':
        training_input = torch.transpose(training_input, -2, -1)

    return training_input

