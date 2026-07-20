from torch.utils.data import DataLoader
from pathlib import Path
import torch
from torch.utils.data import Subset

class DataLoaders:
    def __init__(
        self,
        dataset_path: Path,
        subj_path: Path,
        prompt_path: Path,
        batch_size=64,
        shuffle_train=True,
    ):
        self.dataset = torch.load(dataset_path, weights_only=False)
        self.subj_splits = torch.load(subj_path, weights_only=False)
        self.prompt_splits = torch.load(prompt_path, weights_only=False)
        self.subj_dataset = {
            "train_dataset": Subset(self.dataset, self.subj_splits["train"]),
            "test_dataset": Subset(self.dataset, self.subj_splits["test"]),
            "validate_dataset": Subset(self.dataset, self.subj_splits["validate"]),
        }
        self.prompt_dataset = {
            "train_dataset": Subset(self.dataset, self.prompt_splits["train"]),
            "test_dataset": Subset(self.dataset, self.prompt_splits["test"]),
            "validate_dataset": Subset(self.dataset, self.prompt_splits["validate"]),
        }

        self.subj_loader = self._make_loaders(
            self.subj_dataset, batch_size, shuffle_train
        )
        self.prompt_loader = self._make_loaders(
            self.prompt_dataset, batch_size, shuffle_train
        )

        self.subj_iter = {
            "train": iter(self.subj_loader["train"]),
            "test": iter(self.subj_loader["test"]),
            "validate": iter(self.subj_loader["validate"]),
        }

        self.prompt_iter = {
            "train": iter(self.prompt_loader["train"]),
            "test": iter(self.prompt_loader["test"]),
            "validate": iter(self.prompt_loader["validate"]),
        }

    def _make_loaders(self, datasets, batch_size, shuffle_train):
        return {
            "train": DataLoader(
                datasets["train_dataset"], batch_size=batch_size, shuffle=shuffle_train
            ),
            "test": DataLoader(
                datasets["test_dataset"], batch_size=batch_size, shuffle=False
            ),
            "validate": DataLoader(
                datasets["validate_dataset"], batch_size=batch_size, shuffle=False
            ),
        }

    def next(self, loader: str, subset: str):
        if loader == "s":
            loaders = self.subj_loader
            iterator = self.subj_iter
        elif loader == "p":
            loaders = self.prompt_loader
            iterator = self.prompt_iter
        else:
            raise ValueError(
                "There is no such loader. Did you mean 's' for subject-wise or 'p' for prompt-wise?"
            )
        if subset not in ("train", "test", "validate"):
            raise ValueError(
                f"Expected 'train', 'test', or 'validate' subset. Got '{subset}' :("
            )
        try:
            return next(iterator[subset])
        except StopIteration:
            iterator[subset] = iter(loaders[subset])
            return next(iterator[subset])
