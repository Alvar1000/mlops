from collections.abc import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset


class VariantPairDataset(Dataset):
    def __init__(
        self,
        variant_pairs: Iterable[tuple[int, int]],
        text_embeddings: dict[int, np.ndarray],
        img_embeddings: dict[int, np.ndarray],
        targets: np.ndarray,
    ) -> None:
        self.variant_pairs = variant_pairs
        self.text_embeddings = text_embeddings
        self.img_embeddings = img_embeddings
        self.targets = targets

    def __len__(self) -> int:
        return len(self.variant_pairs)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        variantid1, variantid2 = self.variant_pairs[idx]

        text_emb1 = self.text_embeddings[variantid1]
        img_emb1 = self.img_embeddings[variantid1]
        text_emb2 = self.text_embeddings[variantid2]
        img_emb2 = self.img_embeddings[variantid2]

        target = self.targets[idx]

        sample = {
            "text_emb1": torch.tensor(text_emb1, dtype=torch.float32),
            "img_emb1": torch.tensor(img_emb1, dtype=torch.float32),
            "text_emb2": torch.tensor(text_emb2, dtype=torch.float32),
            "img_emb2": torch.tensor(img_emb2, dtype=torch.float32),
            "target": torch.tensor(target, dtype=torch.float32),
        }

        return sample
