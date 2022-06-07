from torch.nn import LSTM, Linear, Module


class EDLLSTM(Module):
    def __init__(self,
                 spatial_dim,
                 hidden_dim,
                 output_dim,
                 nb_layer,
                 **kwargs):
        super().__init__()
        self.rnn = LSTM(input_size=spatial_dim,
                        hidden_size=hidden_dim,
                        batch_first=True,
                        num_layers=nb_layer)

        self.linear = Linear(in_features=hidden_dim,
                             out_features=output_dim[0])

    def forward(self, x):
        output, _ = self.rnn(x)

        logits = self.linear(output[:, -1, :])

        return logits

