import torch
from torch.utils.data import Dataset, DataLoader


class _AugDataset(Dataset):
    """Aplica augmentations amostra a amostra (só no treino)."""
    def __init__(self, X, y, augs):
        self.X, self.y, self.augs = X, y, augs

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        x = self.X[i]
        for aug in self.augs:
            x = aug(x)
        return x, self.y[i]


class Data:
    def __init__(self, X, y=None, batch_size: int = 32):
        self.batch_size = batch_size
        if isinstance(X, DataLoader):
            self.batch_size = X.batch_size or batch_size
            xs, ys = [], []
            for xb, yb in X:
                xs.append(xb)
                ys.append(yb)
            X, y = torch.cat(xs), torch.cat(ys)
        if y is None:
            raise ValueError("Passe y junto com X, ou um DataLoader.")
        if len(X) != len(y):
            raise ValueError(f"X tem {len(X)} amostras e y tem {len(y)}.")
        self.X, self.y = X, y
        self._steps = []      # pipeline reaplicável em outros conjuntos
        self._augs = []       # augmentations aplicadas só no treino

    def __len__(self):
        return len(self.X)

    def __repr__(self):
        return (f"Data(n={len(self)}, X={tuple(self.X.shape)} {self.X.dtype}, "
                f"y={tuple(self.y.shape)} {self.y.dtype})")

    # ------------------------------------------------ pipeline
    def _do(self, fn):
        self.X, self.y = fn(self.X, self.y)
        self._steps.append(fn)
        return self

    def apply(self, other: "Data"):
        """Repete no `other` (teste/validação) o pipeline já ajustado neste conjunto."""
        for fn in self._steps:
            other.X, other.y = fn(other.X, other.y)
        return other

    # ------------------------------------------------ transformações de X
    def scale(self, factor: float = 255.0):
        return self._do(lambda X, y: (X.float() / factor, y))

    def add_channel(self):
        """(N,H,W) -> (N,1,H,W)"""
        return self._do(lambda X, y: (X.unsqueeze(1) if X.ndim == 3 else X, y))

    def flatten(self):
        return self._do(lambda X, y: (X.flatten(1), y))

    def reshape(self, *shape):
        return self._do(lambda X, y: (X.reshape(X.shape[0], *shape), y))

    def normalize(self, mean=None, std=None):
        """Padroniza por canal (4D) ou por feature (2D). Sem argumentos, ajusta neste conjunto."""
        X = self.X.float()
        dims = (0, 2, 3) if X.ndim == 4 else (0,)
        if mean is None:
            mean = X.mean(dim=dims, keepdim=True)
        else:
            mean = torch.as_tensor(mean, dtype=torch.float32).reshape(1, -1, *([1] * (X.ndim - 2)))
        if std is None:
            std = X.std(dim=dims, keepdim=True)
        else:
            std = torch.as_tensor(std, dtype=torch.float32).reshape(1, -1, *([1] * (X.ndim - 2)))
        std = std.clamp_min(1e-8)
        return self._do(lambda X, y: ((X.float() - mean) / std, y))

    def minmax(self, low: float = 0.0, high: float = 1.0):
        X = self.X.float()
        dims = tuple(range(X.ndim))[:1] if X.ndim == 2 else (0, 2, 3)
        mn = X.amin(dim=dims, keepdim=True)
        mx = X.amax(dim=dims, keepdim=True)
        rng = (mx - mn).clamp_min(1e-8)
        return self._do(lambda X, y: ((X.float() - mn) / rng * (high - low) + low, y))

    def clip(self, low, high):
        return self._do(lambda X, y: (X.clamp(low, high), y))

    # ------------------------------------------------ transformações de y
    def to_long(self):
        return self._do(lambda X, y: (X, y.long()))

    def one_hot(self, num_classes: int | None = None):
        n = num_classes or int(self.y.max().item()) + 1
        return self._do(lambda X, y: (X, torch.nn.functional.one_hot(y.long(), n).float()))

    def encode_labels(self):
        """Mapeia rótulos arbitrários para 0..K-1 (guarda o mapa em self.classes)."""
        self.classes = torch.unique(self.y)
        classes = self.classes
        return self._do(lambda X, y: (X, torch.searchsorted(classes, y)))

    # ------------------------------------------------ amostras
    def shuffle(self, seed: int | None = None):
        g = torch.Generator().manual_seed(seed) if seed is not None else None
        idx = torch.randperm(len(self), generator=g)
        self.X, self.y = self.X[idx], self.y[idx]
        return self

    def subset(self, n: int, seed: int | None = None):
        """Usa só n amostras (bom para testar rápido)."""
        g = torch.Generator().manual_seed(seed) if seed is not None else None
        idx = torch.randperm(len(self), generator=g)[:n]
        self.X, self.y = self.X[idx], self.y[idx]
        return self

    def _child(self, idx):
        d = Data(self.X[idx], self.y[idx], self.batch_size)
        d._steps = list(self._steps)
        d._augs = list(self._augs)
        return d

    def split(self, val: float = 0.2, test: float = 0.0, seed: int | None = None, stratify: bool = False):
        """Devolve (train, val) ou (train, val, test). O pipeline é herdado pelas partes."""
        n = len(self)
        g = torch.Generator().manual_seed(seed) if seed is not None else None

        if stratify and self.y.ndim == 1:
            groups = [torch.nonzero(self.y == c).flatten() for c in torch.unique(self.y)]
            groups = [gi[torch.randperm(len(gi), generator=g)] for gi in groups]
            tr, va, te = [], [], []
            for gi in groups:
                n_te, n_va = int(len(gi) * test), int(len(gi) * val)
                te.append(gi[:n_te]); va.append(gi[n_te:n_te + n_va]); tr.append(gi[n_te + n_va:])
            tr, va, te = torch.cat(tr), torch.cat(va), torch.cat(te)
        else:
            idx = torch.randperm(n, generator=g)
            n_te, n_va = int(n * test), int(n * val)
            te, va, tr = idx[:n_te], idx[n_te:n_te + n_va], idx[n_te + n_va:]

        parts = [self._child(tr), self._child(va)]
        if test > 0:
            parts.append(self._child(te))
        return tuple(parts)

    # ------------------------------------------------ data augmentation (só treino)
    def random_flip(self, p: float = 0.5, horizontal: bool = True):
        dim = -1 if horizontal else -2
        self._augs.append(lambda x: torch.flip(x, dims=[dim]) if torch.rand(()) < p else x)
        return self

    def random_noise(self, std: float = 0.05):
        self._augs.append(lambda x: x + torch.randn_like(x) * std)
        return self

    def random_shift(self, max_shift: int = 2):
        def aug(x):
            dy, dx = torch.randint(-max_shift, max_shift + 1, (2,)).tolist()
            return torch.roll(x, shifts=(dy, dx), dims=(-2, -1))
        self._augs.append(aug)
        return self

    # ------------------------------------------------ saída
    def loader(self, batch_size: int | None = None, shuffle: bool = True,
               augment: bool = True, **kwargs) -> DataLoader:
        augs = self._augs if augment else []
        ds = _AugDataset(self.X, self.y, augs)
        return DataLoader(ds, batch_size=batch_size or self.batch_size, shuffle=shuffle, **kwargs)

    def tensors(self):
        return self.X, self.y

    def summary(self):
        print(self)
        if self.y.ndim == 1 and not self.y.is_floating_point():
            vals, counts = torch.unique(self.y, return_counts=True)
            print("classes:", dict(zip(vals.tolist(), counts.tolist())))
        print(f"X: min={self.X.min():.3f} max={self.X.max():.3f} "
              f"mean={self.X.float().mean():.3f} std={self.X.float().std():.3f}")