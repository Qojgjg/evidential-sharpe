from torch.utils.data import Dataset

from src.data.preprocessing.data_processing import *


class LongShortDataset(Dataset):
    def __init__(self,
                 data_dir,
                 index_range,
                 nb_total_feature,
                 lb_wn_size,
                 no_wn_size,
                 feature_dict,
                 dimension_order,
                 add_channel_dim=False,
                 normalization_scheme="raw",
                 **kwargs):
        assert feature_dict["close"]  # close price must be included to calculate returns.
        assert normalization_scheme in ["raw", "z_score1", "z_score_block"]
        assert dimension_order in ['temporal-spatial', 'spatial-temporal']

        super(LongShortDataset, self).__init__()

        self.data = load_data(data_dir, index_range)
        assert self.data.shape[0] == index_range[1] - index_range[0], "Check raw data range"
        assert self.data.shape[1] == 5, "Check raw data feature size"

        data_selected_feature, self.future_return = normalize_and_return_calculation(self.data,
                                                                                     normalization_scheme,
                                                                                     no_wn_size,
                                                                                     lb_wn_size,
                                                                                     feature_dict)

        self.training_input = make_window(self.data,
                                          data_selected_feature,
                                          no_wn_size,
                                          lb_wn_size,
                                          dimension_order,
                                          add_channel_dim)

        assert self.training_input.shape[0] == (index_range[1] - index_range[0] - 1) - no_wn_size - lb_wn_size, \
            "Check number of windows"
        # assert self.training_input.shape[1] == lb_wn_size, "Check temporal dimension size"
        # assert self.training_input.shape[2] == nb_total_feature, "Check spatial dimension size"

        self.future_return = torch.cat([self.future_return, -self.future_return], dim=1)
        assert self.future_return.shape == (self.training_input.shape[0], 2)

        self.labels = generate_label(future_return=self.future_return)
        assert self.labels.shape == self.future_return.shape

    def __len__(self) -> int:
        return self.training_input.shape[0]

    def __getitem__(self, item):
        return self.training_input[item], \
               self.future_return[item], \
               self.labels[item]

    def __get_future_return__(self):
        return self.future_return

    def __get_whole_dataset__(self):
        return self.training_input

    def __get_class_percentage__(self):
        return self.labels.float().mean(dim=0)
