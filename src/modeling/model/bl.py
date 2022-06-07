import torch
import torch.nn as nn

from src.modeling.custom_layers.bl_layer import BilinearLayer


class BilinearModel(nn.Module):
    def __init__(self,
                 spatial_dim,
                 temporal_dim,
                 topology,
                 output_dim,
                 **kwargs):
        super(BilinearModel, self).__init__()

        input_dim = [spatial_dim, temporal_dim]

        self.all_layers = nn.ModuleList()
        for i in range(len(topology)):
            hidden_shape = topology[i]
            self.all_layers.append(BilinearLayer(input_dim, hidden_shape))
            self.all_layers.append(nn.ReLU())
            input_dim = hidden_shape

        self.all_layers.append(BilinearLayer(input_dim, output_dim))
        self.all_layers.append(nn.Softmax(dim=1))

    def forward(self, x):
        for layer in self.all_layers:
            x = layer(x)

        if x.shape[-1] == 1:
            x = torch.squeeze(x, -1)

        return x
