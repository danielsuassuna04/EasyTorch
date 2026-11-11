import torch
from torch import nn


class Dense(nn.Module):
    def __init__(self, neurons: int = 10, activation = "relu"):
        super().__init__()
        self.neurons = neurons
        self.linear = None
        
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.Identity()

    def build_layer(self, x: torch.Tensor) -> torch.Tensor:
        in_features = x.shape[1]
        self.linear = nn.Linear(in_features, self.neurons)
        return self.activation(self.linear(x))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.activation(self.linear(X))
    
    
class Conv2d(nn.Module):
    def __init__(self, out_channels, kernel_size, stride=1, padding=0, activation="relu"):
        super().__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.conv = None

        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.Identity()

    def build_layer(self, x):
        self.conv = nn.Conv2d(
            x.shape[1], self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
        ).to(x.device)
        return self.activation(self.conv(x))

    def forward(self, X):
        return self.activation(self.conv(X))
    
class BatchNorm2d(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = None

    def build_layer(self, x):
        self.bn = nn.BatchNorm2d(x.shape[1]).to(x.device)
        return self.bn(x)

    def forward(self, X):
        return self.bn(X)
    
class ResidualBlock(nn.Module):
    def __init__(self, out_channels, stride=1):
        super().__init__()
        self.out_channels = out_channels
        self.stride = stride

        self.conv1 = Conv2d(out_channels, 3, stride=stride, padding=1, activation="linear")
        self.bn1 = BatchNorm2d()
        self.conv2 = Conv2d(out_channels, 3, stride=1, padding=1, activation="linear")
        self.bn2 = BatchNorm2d()
        self.shortcut_conv = None
        self.shortcut_bn = None
        self.relu = nn.ReLU()

    def build_layer(self, x):
        in_channels = x.shape[1]
        out = self.bn1.build_layer(self.conv1.build_layer(x))
        out = self.relu(out)
        out = self.bn2.build_layer(self.conv2.build_layer(out))
        if self.stride != 1 or in_channels != self.out_channels:
            self.shortcut_conv = Conv2d(self.out_channels, 1, stride=self.stride, activation="linear")
            self.shortcut_bn = BatchNorm2d()
            identity = self.shortcut_bn.build_layer(self.shortcut_conv.build_layer(x))
        else:
            identity = x

        return self.relu(out + identity)

    def forward(self, X):
        out = self.relu(self.bn1(self.conv1(X)))
        out = self.bn2(self.conv2(out))

        identity = X
        if self.shortcut_conv is not None:
            identity = self.shortcut_bn(self.shortcut_conv(X))

        return self.relu(out + identity)