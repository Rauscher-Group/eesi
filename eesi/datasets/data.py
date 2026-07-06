"""Particle dataset and dataloader factory."""
from typing import Tuple
import torch
from torch.utils.data import Dataset, DataLoader


class ParticleDataset(Dataset):
    """Holds (positions, species) tensors, sorted by species."""

    def __init__(self, positions: torch.Tensor, species: torch.Tensor, L: float):
        if positions.dim() != 3:
            raise ValueError(f"positions must be [M, N, d]; got {tuple(positions.shape)}")
        if species.dim() != 2:
            raise ValueError(f"species must be [M, N]; got {tuple(species.shape)}")
        if positions.shape[:2] != species.shape:
            raise ValueError("positions and species must agree on the first two dims")
        species = species.long()
        idx = species.argsort(dim=-1)
        species = torch.gather(species, 1, idx)
        positions = torch.gather(positions, 1, idx.unsqueeze(-1).expand(-1, -1, positions.shape[-1]))
        self.positions = positions.float()
        self.species = species
        self.L = float(L)

    def __len__(self) -> int:
        return self.positions.shape[0]

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.positions[i], self.species[i]


def make_loader(
    dataset: ParticleDataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    drop_last: bool = True,
    pin_memory: bool = False,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=pin_memory,
    )
