import torch
from torch import nn

def _activation(name: str) -> nn.Module:
    return {"relu": nn.ReLU(), "tanh": nn.Tanh(), "sigmoid": nn.Sigmoid()}.get(name, nn.Identity())


class Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = None                      # módulo criado no build
    def _build(self, x: torch.Tensor):
        """Cria self.layer a partir do shape de x. Camadas sem parâmetros não precisam sobrescrever."""
        pass

    def build_layer(self, x: torch.Tensor) -> torch.Tensor:
        self._build(x)
        return self.forward(x)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.layer(X)


class Dense(Layer):
    def __init__(self, neurons: int = 10, activation="relu"):
        super().__init__()
        self.neurons = neurons
        self.activation = _activation(activation)

    def _build(self, x):
        self.layer = nn.Sequential(nn.Linear(x.shape[1], self.neurons), self.activation)


class Conv2d(Layer):
    def __init__(self, out_channels, kernel_size, stride=1, padding=0, activation="relu"):
        super().__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.activation = _activation(activation)

    def _build(self, x):
        self.layer = nn.Sequential(
            nn.Conv2d(x.shape[1], self.out_channels, self.kernel_size,
                      stride=self.stride, padding=self.padding),
            self.activation,
        )


class BatchNorm2d(Layer):
    def _build(self, x):
        self.layer = nn.BatchNorm2d(x.shape[1])


class MaxPool2D(Layer):
    def __init__(self, kernel_size: int = 2, stride: int | None = None):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride

    def _build(self, x):
        self.layer = nn.MaxPool2d(self.kernel_size, stride=self.stride)


class ReLU(Layer):
    def forward(self, X):
        return torch.relu(X)


class GlobalAvgPool2d(Layer):
    def forward(self, X):
        return X.mean(dim=(2, 3))              # (N, C, H, W) -> (N, C)


class Flatten(Layer):
    def forward(self, X):
        return X.flatten(1)                    # mantém o batch


class Softmax(Layer):
    def forward(self, X):
        return torch.softmax(X, dim=1)


class ResidualBlock(Layer):
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

    def _build(self, x):
        out = torch.relu(self.bn1.build_layer(self.conv1.build_layer(x)))
        self.bn2.build_layer(self.conv2.build_layer(out))

        # projeção 1x1 quando muda resolução ou número de canais
        if self.stride != 1 or x.shape[1] != self.out_channels:
            self.shortcut_conv = Conv2d(self.out_channels, 1, stride=self.stride, activation="linear")
            self.shortcut_bn = BatchNorm2d()
            self.shortcut_bn.build_layer(self.shortcut_conv.build_layer(x))

    def forward(self, X):
        out = torch.relu(self.bn1(self.conv1(X)))
        out = self.bn2(self.conv2(out))

        identity = X
        if self.shortcut_conv is not None:
            identity = self.shortcut_bn(self.shortcut_conv(X))

        self.layer = torch.relu(out + identity)