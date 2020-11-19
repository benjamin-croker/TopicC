from typing import List
import json

import torch
import torch.utils.data


def load_summaries(filename) -> List[str]:
    with open(filename, encoding='utf-8') as f:
        sequences = [s.replace('\n', '') for s in f.readlines()]
    return sequences


def load_categories(filename) -> List[str]:
    with open(filename, encoding='utf-8') as f:
        categories = [s.replace('\n', '') for s in f.readlines()]
    return categories


def load_category_labels(filename) -> dict:
    with open(filename, encoding='utf-8') as f:
        category_labels = json.load(f)
    return category_labels


class WikiVALvl5Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 summaries_fn: str,
                 categories_fn: str,
                 category_labels_fn: str):
        print("init WikiVALvl5Dataset...")
        self.category_to_label_map = load_category_labels(category_labels_fn)
        self.summaries = load_summaries(summaries_fn)
        self.categories = load_categories(categories_fn)

        # remove any empty sequences, or ones with no label mapping
        self.summaries, self.categories = zip(*[
            (summary, category) for summary, category in zip(self.summaries, self.categories)
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
        return self.summaries[index], self.labels[index]

    def labels_to_categories(self, labels: List[int]) -> List[str]:
        return [self.label_to_category_map[label] for label in labels]


def train_test_split(wiki_va_l5_dataset: WikiVALvl5Dataset,
                     test_prop: float = 0.2, seed: int = None):
    n_test = int(len(wiki_va_l5_dataset) * test_prop)
    n_train = len(wiki_va_l5_dataset) - n_test

    if seed is not None:
        generator = torch.Generator().manual_seed(seed)
    else:
        generator = None

    return torch.utils.data.random_split(
        wiki_va_l5_dataset, [n_train, n_test], generator
    )
