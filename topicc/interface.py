import uuid
import os
from typing import List

import torch
from topicc import dataset, model, optimiser
from topicc.model import _TopicCBase

MODEL_LOOKUP = {
    'TopicCDenseSpacy': model.TopicCDenseSpacy,
    'TopicCEncSimpleBPemb': model.TopicCEncSimpleBPemb,
    'TopicCEncBPemb': model.TopicCEncBPemb,
    'TopicKeyEncBPemb': model.TopicKeyEncBPemb,
}
DATASET_LOOKUP = {
    'TopicCDenseSpacy': dataset.SeqCategoryDataset,
    'TopicCEncSimpleBPemb': dataset.SeqCategoryDataset,
    'TopicCEncBPemb': dataset.SeqCategoryDataset,
    'TopicKeyEncBPemb': dataset.SeqKeywordsDataset
}
EVAL_FN_LOOKUP = {
    'TopicCDenseSpacy': optimiser.evaluate_topicc_model,
    'TopicCEncSimpleBPemb': optimiser.evaluate_topicc_model,
    'TopicCEncBPemb': optimiser.evaluate_topicc_model,
    'TopicKeyEncBPemb': optimiser.evaluate_topickey_model
}
EVAL_SCORE_LOOKUP = {
    'TopicCDenseSpacy': 'accuracy',
    'TopicCEncSimpleBPemb': 'accuracy',
    'TopicCEncBPemb': 'accuracy',
    'TopicKeyEncBPemb': 'Jaccard score'
}

CHECKPOINT_DIR = 'checkpoints'
OUTPUT_DIR = 'output'
CPU_DEVICE = 'cpu'


class TopicC(object):
    def __init__(self,
                 model_type: str, model_params: dict, model_state_dict: dict,
                 category_to_label_map: dict):
        self.model = MODEL_LOOKUP[model_type](**model_params)
        self.model.load_state_dict(model_state_dict)
        self.label_to_category_map = {
            category_to_label_map[c]: c for c in category_to_label_map
        }

    def predict(self, sequence, top_k=2) -> List[str]:
        if len(sequence) == 0:
            return []
        log_prob = self.model([sequence])
        preds = self.model.predict(log_prob, top_k)
        return [self.label_to_category_map[int(pred)] for pred in preds.tolist()]


def save_topicc(params: dict, topicc_model: _TopicCBase, filename):
    torch.save(
        {
            'params': params,
            'state_dict': topicc_model.state_dict()
        },
        filename
    )


def load_topicc(filename: str) -> TopicC:
    topicc_spec = torch.load(filename)
    topicc_model = TopicC(
        model_type=topicc_spec['params']['model_type'],
        model_params=topicc_spec['params']['model_params'],
        model_state_dict=topicc_spec['state_dict'],
        category_to_label_map=dataset.load_category_labels(
            topicc_spec['params']['dataset_params']['category_labels_file']
        )
    )
    return topicc_model


def train_topicc(params: dict):
    model_id = params.get('model_id')
    if model_id is None:
        model_id = str(uuid.uuid1())
        params['model_id'] = model_id

    print(f'start: {model_id}')
    topicc_model = MODEL_LOOKUP[params['model_type']](**params['model_params'])
    topicc_dataset = DATASET_LOOKUP[params['model_type']](**params['dataset_params'])

    # make sure the output directories exist
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    topicc_model = optimiser.train(
        topicc_model, topicc_dataset,
        model_id, CHECKPOINT_DIR,
        eval_fn=EVAL_FN_LOOKUP[params['model_type']],
        score_name=EVAL_SCORE_LOOKUP[params['model_type']],
        **params['optimiser_params']
    )
    save_topicc(
        params, topicc_model, os.path.join(OUTPUT_DIR, f'{model_id}.topicc')
    )

    print('done')
