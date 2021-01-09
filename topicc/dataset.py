from typing import List, Union, Tuple
import json

import torch
import torch.utils.data
import numpy as np


def _load_text_file(filename) -> List[str]:
    with open(filename, encoding='utf-8') as f:
        data = [s.replace('\n', '') for s in f.readlines()]
    return data


def _load_json_file(filename) -> dict:
    with open(filename, encoding='utf-8') as f:
        data = json.load(f)
    return data


# application-specific names for external use
load_sequences = _load_text_file
load_categories = _load_text_file
load_keywords = _load_text_file
load_category_labels = _load_json_file


class SeqCategoryDataset(torch.utils.data.Dataset):
    def __init__(self,
                 sequences_file: str,
                 categories_file: str,
                 category_labels_file: str):
        print("init: SeqCategoryDataset")
        self.category_to_label_map = load_category_labels(category_labels_file)
        self.sequences = load_sequences(sequences_file)
        self.categories = load_categories(categories_file)

        # remove any empty sequences, or ones with no label mapping
        self.sequences, self.categories = zip(*[
            (summary, category) for summary, category in zip(self.sequences, self.categories)
            if len(summary) > 0 and category in self.category_to_label_map
        ])

        self.label_to_category_map = {
            self.category_to_label_map[c]: c for c in self.category_to_label_map
        }
        self.labels = [self.category_to_label_map[c] for c in self.categories]
        self.n_labels = len(self.category_to_label_map)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        # note that dataloader will return a tuple of strings for the sequences
        # and a 1-D float tensor for the labels in each batch
        return self.sequences[index], self.labels[index]

    def labels_to_categories(self, labels: List[int]) -> List[str]:
        return [self.label_to_category_map[label] for label in labels]


class SeqKeywordsDataset(torch.utils.data.Dataset):
    def __init__(self,
                 sequences_file: str,
                 keywords_file: str,
                 # may need a max len to prevent issues with CUDA
                 max_len = 1024):
        print("init: SeqKeywordsDataset")
        self.sequences = load_sequences(sequences_file)
        self.keywords = load_keywords(keywords_file)
        # convert to a set for fast lookups and duplicate removal
        self.keywords = [set(keyword.lower().split()) for keyword in self.keywords]

        self.max_len = max_len
        # TODO: consider removing stopwords
    
    @staticmethod
    def collate_fn(batch):
        seqs = [np.array(item[0]) for item in batch]
        labels = [torch.tensor(item[1], dtype=torch.int) for item in batch]
        return [seqs, labels]

    def __len__(self):
        return len(self.keywords)

    def __getitem__(self, index) -> Tuple[torch.tensor, torch.tensor]:
        seq = self.sequences[index].lower().split()[0:self.max_len]
        labels = [word in self.keywords[index] for word in seq]
        return seq, labels


def train_test_split(dataset: Union[SeqCategoryDataset, SeqKeywordsDataset],
                     test_prop: float = 0.2, seed: int = None):
    n_test = int(len(dataset) * test_prop)
    n_train = len(dataset) - n_test

    if seed is not None:
        generator = torch.Generator().manual_seed(seed)
    else:
        generator = None

    return torch.utils.data.random_split(
        dataset, [n_train, n_test], generator
    )
