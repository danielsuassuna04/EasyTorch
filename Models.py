from torch import nn
import torch
import Layer


class Model(nn.Module):

    def __init__(self):
        super().__init__()

    def compile(self,optimizer = "Adam", loss = "mse", lr: float = 0.05):

        if loss == "mse":
            self.loss = nn.L1Loss()
        elif loss == "crossentropy":
            self.loss = nn.CrossEntropyLoss()
        else:
            raise ValueError("Valor não suportado para loss")

        if optimizer == "Adam":
            self.optimizer = torch.optmim.Adam(self.parameters(), lr = lr)


    def train(self,X: torch.Tensor, epochs:int):
        return 
class Sequential(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList()
        self._built = False

    def add(self, layer):
        self.layers.append(layer)

    def build(self, input_shape):
        if self._built:
            return
        x = torch.zeros(1, *input_shape)
        
        for layer in self.layers:
            if hasattr(layer, 'build_layer'):
                x = layer.build_layer(x)
            else:
                x = layer(x)
                
        self._built = True

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if not self._built:
            self.build(X.shape[1:])
        
        for layer in self.layers:
            X = layer(X)
        return X
    
class ResNet(Sequential):
    def __init__(self, blocks_per_stage=(2, 2, 2, 2), num_classes=1000,
                 channels=(64, 128, 256, 512)):
        super().__init__()
        self.add(Layer.Conv2d(channels[0], 7, stride=2, padding=3, activation="linear"))
        self.add(Layer.BatchNorm2d())
        self.add(nn.ReLU())
        self.add(nn.MaxPool2d(3, stride=2, padding=1))

        # estágios residuais
        for i, (n_blocks, ch) in enumerate(zip(blocks_per_stage, channels)):
            for j in range(n_blocks):
                stride = 2 if (j == 0 and i > 0) else 1  # reduz resolução no 1º bloco de cada estágio (exceto o primeiro)
                self.add(Layer.ResidualBlock(ch, stride=stride))

        # cabeça
        self.add(nn.AdaptiveAvgPool2d((1, 1)))
        self.add(nn.Flatten())
        self.add(Layer.Dense(num_classes, activation="linear"))




    