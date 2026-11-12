import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm.auto import tqdm
import Layer
from DataTransform import Data

# ---------------------------------------------------------------- device
_DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def device(name: str | None = None) -> torch.device:
    """device() devolve o padrão; device("cuda") ou device("cpu") altera o padrão global."""
    global _DEFAULT_DEVICE
    if name is not None:
        _DEFAULT_DEVICE = torch.device(name)
    return _DEFAULT_DEVICE


# ---------------------------------------------------------------- camada
class Softmax(nn.Module):
    def forward(self, X):
        return torch.softmax(X, dim=1)


# ---------------------------------------------------------------- Model
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.batch_size = 32
        self.loss_func = None
        self.optimizer = None
        self._opt_cfg = None
        self._ready = False
        self.device = device()

    # ---------- configuração ----------
    def compile(self, optimizer="adam", loss="crossentropy", lr=1e-3, batch_size=32,
            weight_decay=0.0, label_smoothing=0.0, scheduler=None,
            grad_clip=None, amp=False):
        self.batch_size = batch_size

        loss = loss.lower()
        if loss == "mse":
            self.loss_func = nn.MSELoss()
        elif loss == "crossentropy":
            self.loss_func = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        else:
            raise ValueError("Valor não suportado para loss")

        optimizer = optimizer.lower()
        if optimizer not in ("adam", "adamw", "sgd"):
            raise ValueError("Otimizador não suportado")

        self._opt_cfg = (optimizer, lr, weight_decay)
        self._scheduler_name = scheduler
        self._scheduler = None
        self.grad_clip = grad_clip
        self._amp = amp
        self.optimizer = None
        self._ready = False

    def _make_optimizer(self):
        name, lr, wd = self._opt_cfg
        decay, no_decay = [], []
        for n, p in self.named_parameters():
            (no_decay if p.ndim <= 1 or n.endswith(("cls", "pe")) else decay).append(p)
        groups = [{"params": decay, "weight_decay": wd},
                {"params": no_decay, "weight_decay": 0.0}]
        opt_cls = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW, "sgd": torch.optim.SGD}[name]
        return opt_cls(groups, lr=lr)

    def set_device(self, dev: str):
        self.device = torch.device(dev)
        if self._ready:
            self.to(self.device)
            self.optimizer = self._make_optimizer()
            self._scaler = torch.amp.GradScaler(enabled=self._amp and self.device.type == "cuda")
            self._ready = True   # estado do otimizador ficou no device antigo
        return self

    def _prepare(self, sample_inputs):
        """Constrói camadas lazy, move para o device e cria o otimizador."""
        if self._ready:
            return
        self.eval()
        with torch.no_grad():
            self(sample_inputs[:1])

        self.to(self.device)
        self.optimizer = self._make_optimizer()
        self._ready = True

    # ---------- forward para a loss ----------
    def _forward_logits(self, X):
        """Saída usada na loss. Modelos sem Softmax final usam o forward normal."""
        return self(X)

    # ---------- utilitários ----------
    def _to_loader(self, data, shuffle):
        if data is None:
            return None
        if isinstance(data, Data):
            return data.loader(batch_size=self.batch_size, shuffle=shuffle, augment=shuffle)
        if isinstance(data, DataLoader):
            return data
        if isinstance(data, (tuple, list)) and len(data) == 2:
            X, y = data
            return DataLoader(TensorDataset(X, y), batch_size=self.batch_size, shuffle=shuffle)
        raise TypeError("Passe um Data, um DataLoader ou uma tupla (X, y)")

    def _compute_metrics(self, preds, targets, metrics):
        out = {}
        for m in metrics:
            m = m.lower()
            if m == "accuracy":
                pred_labels = preds.argmax(dim=1)
                true_labels = targets.argmax(dim=1) if targets.ndim > 1 and targets.size(1) > 1 else targets
                out["Acurácia"] = f"{(pred_labels == true_labels).float().mean().item() * 100:.2f}%"
            elif m == "mae":
                out["MAE"] = f"{torch.mean(torch.abs(preds - targets)).item():.4f}"
            elif m == "mse":
                out["MSE"] = f"{torch.mean((preds - targets) ** 2).item():.4f}"
            else:
                raise ValueError(f"Métrica não suportada: {m}")
        return out

    # ---------- treino ----------
    def train_step(self, train_loader, test_loader, epoch, epochs, metrics):
        self.train()
        use_amp = self._amp and self.device.type == "cuda"
        epoch_loss = 0.0
        bar = tqdm(train_loader, desc=f"Época {epoch+1}/{epochs}", unit="batch")

        for inputs, targets in bar:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=use_amp):
                loss = self.loss_func(self._forward_logits(inputs), targets)

            self._scaler.scale(loss).backward()
            if self.grad_clip:
                self._scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)
            self._scaler.step(self.optimizer)
            self._scaler.update()
            if self._scheduler is not None:
                self._scheduler.step()

            epoch_loss += loss.item()
            bar.set_postfix(loss=f"{loss.item():.4f}")

        self.eval()
        train_loss = epoch_loss / len(train_loader)
        parts = [f"Loss treino: {train_loss:.4f}"]
        monitored = train_loss

        if test_loader is not None:
            all_preds, all_targets, val_loss, n = [], [], 0.0, 0
            with torch.inference_mode():
                for inputs, targets in test_loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=use_amp):
                        outputs = self._forward_logits(inputs)
                        val_loss += self.loss_func(outputs, targets).item() * targets.size(0)
                    n += targets.size(0)
                    all_preds.append(outputs.float())
                    all_targets.append(targets)

            preds, tgts = torch.cat(all_preds), torch.cat(all_targets)
            monitored = val_loss / max(n, 1)
            parts.append(f"Loss val: {monitored:.4f}")
            parts += [f"{k}: {v}" for k, v in self._compute_metrics(preds, tgts, metrics).items()]

        print(f"Fim da Época {epoch+1}/{epochs} | " + " | ".join(parts))
        return monitored

    def fit(self, train, test=None, epochs=10, metrics=None, save_best: str | None = None):
        if self.loss_func is None:
            raise RuntimeError("Chame compile() antes de fit().")
        metrics = metrics or []

        train_loader = self._to_loader(train, shuffle=True)
        test_loader = self._to_loader(test, shuffle=False)

        sample_inputs, _ = next(iter(train_loader))
        self._prepare(sample_inputs)

        if self._scheduler_name == "onecycle":
            self._scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer, max_lr=self._opt_cfg[1],
                total_steps=epochs * len(train_loader), pct_start=0.1)

        best = float("inf")
        for epoch in range(epochs):
            monitored = self.train_step(train_loader, test_loader, epoch, epochs, metrics)
            if save_best and monitored < best:
                best = monitored
                torch.save(self.state_dict(), save_best)

    # ---------- inferência ----------
    @torch.inference_mode()
    def predict(self, X, batch_size: int | None = None):
        """Devolve a saída completa do modelo (com Softmax, se houver), na CPU."""
        self.eval()
        batch_size = batch_size or self.batch_size
        outs = [self(X[i:i + batch_size].to(self.device)).cpu()
                for i in range(0, len(X), batch_size)]
        return torch.cat(outs)


# ---------------------------------------------------------------- Sequential
class Sequential(Model):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList()
        self._built = False

    def add(self, layer):
        self.layers.append(layer)

    def build(self, input_shape):
        if self._built:
            return
        was_training = self.training
        self.eval()
        with torch.no_grad():
            x = torch.zeros(1, *input_shape)
            for layer in self.layers:
                x = layer.build_layer(x) if hasattr(layer, "build_layer") else layer(x)
        self.train(was_training)
        self._built = True

    def _run(self, X, skip_final_softmax: bool):
        if not self._built:
            self.build(X.shape[1:])

        layers = self.layers
        if skip_final_softmax and len(layers) > 0 and isinstance(layers[-1], Layer.Softmax):
            layers = layers[:-1]
        for layer in layers:
            X = layer(X)
        return X

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self._run(X, skip_final_softmax=False)

    def _forward_logits(self, X):
        return self._run(X, skip_final_softmax=True)


# ---------------------------------------------------------------- ResNet
class ResNet(Sequential):
    def __init__(self, blocks_per_stage=(2, 2, 2, 2), num_classes=10,
                 channels=(32, 64, 128, 256), softmax: bool = True):
        super().__init__()

        # stem para imagens pequenas (28x28): sem stride e sem maxpool
        self.add(Layer.Conv2d(channels[0], 3, stride=1, padding=1, activation="linear"))
        self.add(Layer.BatchNorm2d())
        self.add(Layer.ReLU())

        for i, (n_blocks, ch) in enumerate(zip(blocks_per_stage, channels)):
            for j in range(n_blocks):
                stride = 2 if (j == 0 and i > 0) else 1
                self.add(Layer.ResidualBlock(ch, stride=stride))

        self.add(Layer.GlobalAvgPool2d())
        self.add(Layer.Flatten())
        self.add(Layer.Dense(num_classes, activation="linear"))
        if softmax:
            self.add(Layer.Softmax())
            
class VisionTransformer(Sequential):
    def __init__(self, patch_size=4, embed_dim=64, depth=4, num_heads=4,
                 mlp_dim=128, num_classes=10, dropout=0.1, softmax: bool = True):
        super().__init__()

        # (N, C, H, W) -> (N, L, E)
        self.add(Layer.PatchEmbedding(embed_dim, patch_size))
        self.add(Layer.ClassToken())
        self.add(Layer.PositionalEncoding())
        self.add(Layer.Dropout(dropout))

        for _ in range(depth):
            self.add(Layer.TransformerEncoderBlock(num_heads, mlp_dim, dropout))

        self.add(Layer.LayerNorm())
        self.add(Layer.TakeClassToken())               # (N, L, E) -> (N, E)
        self.add(Layer.Dense(num_classes, activation="linear"))
        if softmax:
            self.add(Layer.Softmax())
            
class CNNViT(Sequential):
    def __init__(self, channels=(32, 64, 128), blocks_per_stage=(1, 1, 1),
                 embed_dim=64, depth=4, num_heads=4, mlp_dim=128,
                 num_classes=10, dropout=0.1, softmax: bool = True):
        super().__init__()

        # stem para imagens pequenas (28x28)
        self.add(Layer.Conv2d(channels[0], 3, stride=1, padding=1, activation="linear"))
        self.add(Layer.BatchNorm2d())
        self.add(Layer.ReLU())

        # backbone CNN: cada estágio (exceto o primeiro) reduz a resolução pela metade
        # 28x28 -> 28x28 -> 14x14 -> 7x7
        for i, (n_blocks, ch) in enumerate(zip(blocks_per_stage, channels)):
            for j in range(n_blocks):
                stride = 2 if (j == 0 and i > 0) else 1
                self.add(Layer.ResidualBlock(ch, stride=stride))

        # cada posição do mapa de features vira um token (7x7 = 49 tokens)
        self.add(Layer.PatchEmbedding(embed_dim, patch_size=1))
        self.add(Layer.ClassToken())
        self.add(Layer.PositionalEncoding())
        self.add(Layer.Dropout(dropout))

        for _ in range(depth):
            self.add(Layer.TransformerEncoderBlock(num_heads, mlp_dim, dropout))

        self.add(Layer.LayerNorm())
        self.add(Layer.TakeClassToken())
        self.add(Layer.Dense(num_classes, activation="linear"))
        if softmax:
            self.add(Layer.Softmax())