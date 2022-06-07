import torch
import torch.nn as nn
import torch.nn.functional as F


def nmodeproduct(x, W, mode):
    assert mode in [1, 2], 'only support mode 1, 2'
    if mode == 1:
        y = torch.transpose(x, 1, 2)  # N x D x T ==> N x T x D.
        y = F.linear(y, W)  # (N x T x D) x (D' x D)' ==> N x T x D'.
        y = torch.transpose(y, 1, 2)  # N x T x D' ==> N x D' x T.
    else:
        y = F.linear(x, W)  # (N x D' x T) x (T' x T)' ==> N x D' x T'.

    return y


class TABLLayer(nn.Module):
    def __init__(self,
                 input_shape,
                 output_shape,
                 device):
        super(TABLLayer, self).__init__()

        self.in1, self.in2 = input_shape
        self.out1, self.out2 = output_shape
        self.device = device

        self.W1 = nn.Parameter(data=torch.Tensor(self.out1, self.in1),
                               requires_grad=True)                          # D' x D.

        self.W = nn.Parameter(data=torch.Tensor(self.in2, self.in2))        # T x T.

        self.W2 = nn.Parameter(data=torch.Tensor(self.out2, self.in2),
                               requires_grad=True)                          # T' x T.

        self.alpha = nn.Parameter(data=torch.Tensor(1,),
                                  requires_grad=True)

        self.bias = nn.Parameter(data=torch.Tensor(self.out1, self.out2),
                                 requires_grad=True)                        # D' x T'.

        # initialization
        nn.init.xavier_uniform_(self.W1)
        nn.init.constant_(self.W, 1/self.in2)
        nn.init.xavier_uniform_(self.W2)
        nn.init.constant_(self.alpha, 0.5)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        # Input shape:
        #   x: N x D x T.
        #   W1: D' x D.
        #   x_bar: N x D' x T.
        x_bar = nmodeproduct(x, self.W1, 1)

        # Create a copy of self.W that diagonal elements are 1/T and other elements are equal self.W.
        # self.W shape: TxT.
        # W shape: TxT.
        W = self.W - self.W * torch.eye(self.in2, device=self.device) + \
            (1/self.in2) * torch.eye(self.in2, device=self.device)

        # Input shape:
        #   x_bar: N x D' x T.
        #   W: T x T.
        # Output shape:
        #   E: N x D' x T.
        E = nmodeproduct(x_bar, W, 2)  # (N x D' x T) x (T x T)' ==> N x D' x T.

        # Taking Softmax over temporal dimension.
        A = F.softmax(E, dim=-1)  # N x D' x T ==> N x D' x T.

        x_tilde = self.alpha * (x_bar * A) + (1 - self.alpha) * x_bar  # N x D' x T ==> N x D' x T.

        # Input shape:
        #   X_bar: N x D' x T.
        #   W2: T x T'.
        # Output shape:
        #   y: N x D' x T'.
        y = nmodeproduct(x_bar, self.W2, 2)

        # bias: D' x T'.
        y = y + self.bias

        return y
