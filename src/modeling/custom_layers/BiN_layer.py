import torch
import torch.nn as nn
import torch.nn.functional as F


class BiN(nn.Module):
    def __init__(self,
                 temporal_dim,
                 spatial_dim,
                 dimension_order,
                 normalization_scheme,
                 epsilon=1e-4):

        super(BiN, self).__init__()

        assert normalization_scheme == "raw", "Used unnormalized dataset for BiN layer"

        self.temporal_dim = temporal_dim
        self.spatial_dim = spatial_dim
        self.dim_ordering = dimension_order
        self.epsilon = epsilon

        assert dimension_order in ['temporal-spatial', 'spatial-temporal']

        if dimension_order == 'temporal-spatial':
            temporal_shape = (1, temporal_dim, 1)
            spatial_shape = (1, 1, spatial_dim)
            self.temporal_axis = 1
            self.spatial_axis = 2
        else:
            temporal_shape = (1, 1, temporal_dim)
            spatial_shape = (1, spatial_dim, 1)
            self.temporal_axis = 2
            self.spatial_axis = 1

        # Below comments are example of 'spatial-temporal' case.
        self.gamma_temporal = nn.Parameter(data=torch.Tensor(temporal_shape[0],
                                                             temporal_shape[1],
                                                             temporal_shape[2]),
                                           requires_grad=True)  # 1 x 1 x T.

        self.beta_temporal = nn.Parameter(data=torch.Tensor(temporal_shape[0],
                                                             temporal_shape[1],
                                                             temporal_shape[2]),
                                           requires_grad=True)  # 1 x 1 x T.

        self.gamma_spatial = nn.Parameter(data=torch.Tensor(spatial_shape[0],
                                                             spatial_shape[1],
                                                             spatial_shape[2]),
                                           requires_grad=True)  # 1 x D x 1.

        self.beta_spatial = nn.Parameter(data=torch.Tensor(spatial_shape[0],
                                                             spatial_shape[1],
                                                             spatial_shape[2]),
                                           requires_grad=True)  # 1 x D x 1.

        self.lambda1 = nn.Parameter(data=torch.Tensor(1,),
                                    requires_grad=True)

        self.lambda2 = nn.Parameter(data=torch.Tensor(1,),
                                    requires_grad=True)

        # initialization
        nn.init.ones_(self.gamma_temporal)
        nn.init.zeros_(self.beta_temporal)
        nn.init.ones_(self.gamma_spatial)
        nn.init.zeros_(self.beta_spatial)
        nn.init.ones_(self.lambda1)
        nn.init.ones_(self.lambda2)

    def forward(self, x):
        # normalize temporal mode
        # N x T x D ==> N x 1 x D or
        # N x D x T ==> N x D x 1.
        tem_mean = torch.mean(x, self.temporal_axis, keepdims=True)

        # N x T x D ==> N x 1 x D or
        # N x D x T ==> N x D x 1.
        tem_std = torch.std(x, self.temporal_axis, keepdims=True)

        # print('mean tem value has nan: %s' % str(torch.isnan(tem_mean).any()))
        # print('std tem value has nan: %s' % str(torch.isnan(tem_mean).any()))

        # mask = tem_std >= self.epsilon
        # tem_std = tem_std*mask + torch.logical_not(mask)*torch.ones(tem_std.size(), requires_grad=False)
        tem_std[tem_std < self.epsilon] = 1.0
        tem = (x - tem_mean) / tem_std

        # N x T x D ==> N x T x 1 or
        # N x D x T ==> N x 1 x T.
        spatial_mean = torch.mean(x, self.spatial_axis, keepdims=True)
        spatial_std = torch.std(x, self.spatial_axis, keepdims=True)

        # print('mean spat value has nan: %s' % str(torch.isnan(spatial_mean).any()))
        # print('std spat value has nan: %s' % str(torch.isnan(spatial_mean).any()))

        # mask = spatial_std >= self.epsilon
        # spatial_std = spatial_std * mask + torch.logical_not(mask)*torch.ones(spatial_std.size(), requires_grad=False)

        spatial_std[spatial_std < self.epsilon] = 1.0
        spat = (x - spatial_mean) / spatial_std

        outputs1 = self.gamma_temporal * tem + self.beta_temporal
        outputs2 = self.gamma_spatial * spat + self.beta_spatial

        return self.lambda1 * outputs1 + self.lambda2 * outputs2

