import torch
import torch.nn as nn

from src.modeling.custom_layers.bl_layer import BilinearLayer
from src.modeling.custom_layers.tabl_layer import TABLLayer


class TABLModel(nn.Module):
    def __init__(self,
                 spatial_dim,
                 temporal_dim,
                 topology,
                 output_dim,
                 device,
                 **kwargs):
        super(TABLModel, self).__init__()

        input_dim = [spatial_dim, temporal_dim]  # TABL uses different dimension order with LSTM.

        self.all_layers = nn.ModuleList()
        for i in range(len(topology)):
            hidden_shape = topology[i]
            self.all_layers.append(BilinearLayer(input_dim, hidden_shape))
            self.all_layers.append(nn.ReLU())
            input_dim = hidden_shape

        self.all_layers.append(TABLLayer(input_dim, output_dim, device=device))

    def forward(self, x):
        for layer in self.all_layers:
            x = layer(x)

        if x.shape[-1] == 1:
            x = torch.squeeze(x, -1)
        # x = x.view(x.size(0), -1)
        return x
