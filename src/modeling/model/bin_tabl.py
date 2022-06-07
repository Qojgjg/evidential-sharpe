import torch
import torch.nn as nn

from src.modeling.custom_layers.BiN_layer import BiN
from src.modeling.custom_layers.bl_layer import BilinearLayer
from src.modeling.custom_layers.tabl_layer import TABLLayer


class BiNTABL(nn.Module):
    def __init__(self,
                 spatial_dim,
                 temporal_dim,
                 topology,
                 output_dim,
                 normalization_scheme,
                 device,
                 **kwargs):
        super(BiNTABL, self).__init__()

        input_dim = [spatial_dim, temporal_dim]

        self.all_layers = nn.ModuleList()
        self.all_layers.append(BiN(temporal_dim=temporal_dim,
                                   spatial_dim=spatial_dim,
                                   dimension_order="spatial-temporal",
                                   normalization_scheme=normalization_scheme))

        for hidden_shape in topology:
            self.all_layers.append(BilinearLayer(input_dim, hidden_shape))
            self.all_layers.append(nn.ReLU())
            input_dim = hidden_shape

        self.all_layers.append(TABLLayer(input_dim, output_dim, device=device))

    def forward(self, x):
        for layer in self.all_layers:
            x = layer(x)

        if x.shape[-1] == 1:
            x = torch.squeeze(x, -1)

        return x

