import torch
from torch import nn

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

empirical_coverage_value_mean = 6.2
resolution = 10
initial_bias = empirical_coverage_value_mean * resolution


class DNAConvNet(nn.Module):
    def __init__(
        self,
        n_features=4,
        output_length=3000 // resolution,  # 3000 bp at 10 bp resolution
        n_filters=32,
        kernel_size=15,
    ):
        super(DNAConvNet, self).__init__()

        # 1D Convolution layer
        self.conv1 = nn.Conv1d(
            in_channels=n_features,
            out_channels=n_filters,
            kernel_size=kernel_size,
            stride=2,
        )

        self.pool = nn.MaxPool2d(kernel_size=(4, 2), stride=(4, 2))

        # Dense layer to reduce to output size
        self.dense = nn.Linear(9968, output_length)
        self.dense.bias.data.fill_(initial_bias)

        self.softplus = nn.Softplus()

        self.to(device)

    def forward(self, x):
        # Input shape: [batch, sequence_length, n_features]
        # Conv1d expects: [batch, n_features, sequence_length]
        x = x.permute(0, 2, 1)

        # Apply convolution
        x = self.conv1(x)

        x = self.pool(x)

        # Flatten
        x = x.reshape(x.size(0), -1)

        # Dense layer
        x = self.dense(x)

        x = self.softplus(x)

        return x
