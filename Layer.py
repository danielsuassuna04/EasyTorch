import torch
from torch import nn

def _activation(name: str) -> nn.Module:
    return {"relu": nn.ReLU(), "gelu": nn.GELU(), "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid()}.get(name, nn.Identity())


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

class DenseTransformer(Layer):
    def __init__(self, neurons: int = 10, activation="relu"):
        super().__init__()
        self.neurons = neurons
        self.activation = _activation(activation)
    def _build(self, x):
        self.layer = nn.Sequential(nn.Linear(x.shape[-1], self.neurons), self.activation)
        
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

class LearnedPositionalEncoding(nn.Module):
    def __init__(self, seq_len: int, embed_dim: int):
        super().__init__()
        self.pe = nn.Parameter(torch.zeros(1, seq_len, embed_dim))
        nn.init.trunc_normal_(self.pe, std=0.02)

    def forward(self, x):                                             # x: (N, L, E)
        return x + self.pe
    
class PositionalEncoding(Layer):
    def _build(self, x):                                              # x: (N, L, E)
        self.layer = LearnedPositionalEncoding(x.shape[1], x.shape[2])

    def forward(self, X):
        return self.layer(X)
        
class MultiHeadSelfAttention(Layer):
    def __init__(self, num_heads, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.dropout = dropout

    def _build(self, x):
        self.layer = nn.MultiheadAttention(
            embed_dim=x.shape[-1],
            num_heads=self.num_heads,
            dropout=self.dropout,
            batch_first=True,
        )

    def forward(self, X):
        out, _ = self.layer(X, X, X, need_weights=False)
        return out

class Dropout(Layer):
    def __init__(self, p):
        super().__init__()
        self.p = p

    def _build(self, x):
        self.layer = nn.Dropout(self.p)

class PatchEmbedding(Layer):
    """(N, C, H, W) -> (N, L, E), com L = (H/p)*(W/p)"""
    def __init__(self, embed_dim, patch_size):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size

    def _build(self, x):
        self.layer = nn.Conv2d(x.shape[1], self.embed_dim,
                               kernel_size=self.patch_size, stride=self.patch_size)

    def forward(self, X):
        return self.layer(X).flatten(2).transpose(1, 2)


class ClassToken(Layer):
    """Concatena um token [CLS] aprendível no início da sequência."""
    def _build(self, x):
        self.cls = nn.Parameter(torch.zeros(1, 1, x.shape[-1]))
        nn.init.trunc_normal_(self.cls, std=0.02)

    def forward(self, X):
        return torch.cat([self.cls.expand(X.shape[0], -1, -1), X], dim=1)


class TakeClassToken(Layer):
    def forward(self, X):
        return X[:, 0]                          # (N, L, E) -> (N, E)


class LayerNorm(Layer):
    def _build(self, x):
        self.layer = nn.LayerNorm(x.shape[-1])
        
class TransformerEncoderBlock(Layer):
    def __init__(self, num_heads, mlp_dim, dropout=0.0):
        super().__init__()
        self.mlp_dim = mlp_dim
        self.ln1 = LayerNorm()
        self.attn = MultiHeadSelfAttention(num_heads, dropout)
        self.ln2 = LayerNorm()
        self.fc1 = Dense(mlp_dim, "gelu")
        self.fc2 = None                         # precisa do embed_dim, criado no build
        self.drop = Dropout(dropout)

    def _build(self, x):                        # x: (N, L, E)
        h = self.attn.build_layer(self.ln1.build_layer(x))
        x = x + self.drop.build_layer(h)

        h = self.fc1.build_layer(self.ln2.build_layer(x))
        self.fc2 = Dense(x.shape[-1], "linear")
        self.fc2.build_layer(h)

    def forward(self, X):
        X = X + self.drop(self.attn(self.ln1(X)))
        X = X + self.drop(self.fc2(self.fc1(self.ln2(X))))
        return X