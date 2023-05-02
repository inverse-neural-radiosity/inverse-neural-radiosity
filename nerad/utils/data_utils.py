from typing import Iterator

from torch.utils.data import DataLoader


class IndexDataset:
    def __init__(self, n: int):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, idx: int):
        return idx


def create_index_loader(n: int, shuffle: bool) -> Iterator:
    loader = DataLoader(IndexDataset(n), shuffle=shuffle)
    while True:
        for batch in loader:
            yield batch
