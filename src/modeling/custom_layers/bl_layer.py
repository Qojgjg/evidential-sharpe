import torch
import torch.nn as nn
import torch.nn.functional as F


def nmodeproduct(x, W, mode):
    assert mode in [1, 2], 'only support mode 1, 2'

    if mode == 1:
        y = torch.transpose(x, 1, 2)  # N x D x T' ==> N x T' x D.
        y = F.linear(y, W)  # (N x T' x D) x (D' x D)' ==> N x T' x D'.
        y = torch.transpose(y, 1, 2)  # N x T' x D' ==> N x D' x T'.
    else:
        y = F.linear(x, W)  # (N x D x T) x (T' x T)' ==> N x D x T'.

    return y


class BilinearLayer(nn.Module):
    def __init__(self, input_shape, output_shape):
        super(BilinearLayer, self).__init__()
        self.in1, self.in2 = input_shape  # D x T.
        self.out1, self.out2 = output_shape  # D' x T'.

        self.W1 = nn.Parameter(data=torch.Tensor(self.out1, self.in1),
                               requires_grad=True)  # D' x D.

        self.W2 = nn.Parameter(data=torch.Tensor(self.out2, self.in2),
                               requires_grad=True)  # T' x T.

        # initialization
        nn.init.xavier_uniform_(self.W1)
        nn.init.xavier_uniform_(self.W2)

    def forward(self, x):

        # 2-mode product
        y1 = nmodeproduct(x, self.W2, 2)  # N x D x T ==> N x D x T'.

        # 1-mode product
        outputs = nmodeproduct(y1, self.W1, 1)  # N x D x T' ==> N x D' x T'.

        return outputs
