import time
import yaml
import torch

import scipy.sparse as sp
import numpy as np
import seaborn as sns
import torch.nn.functional as F

# from tqdm.notebook import tqdm
from torch.optim import Adam, Optimizer
from collections import defaultdict
from torch_geometric.data import Data, InMemoryDataset

from digl.data import get_dataset, PPRDataset, SDRFCDataset, UndirectedDataset, UndirectedPPRDataset, set_train_val_test_split, set_train_val_test_split_webkb
from digl.models import GCN
from digl.seeds import val_seeds, test_seeds

import sherpa

client = sherpa.Client()
trial = client.get_trial()

DATASET = trial.parameters['dataset']
USE_LCC = True
# DEVICE = trial.parameters['device']
DEVICE = f"cuda:{trial.parameters['device']}"
PREPROCESSING = trial.parameters['preprocessing']

if PREPROCESSING == 'none':
    dataset = get_dataset(
        name=DATASET,
        use_lcc=USE_LCC,
    )
if PREPROCESSING == 'undirected':
    dataset = UndirectedDataset(
        name=DATASET,
        use_lcc=USE_LCC,
    )
elif PREPROCESSING == 'ppr':
    dataset = PPRDataset(
        name=DATASET,
        use_lcc=USE_LCC,
        alpha=trial.parameters['alpha'],
        k=trial.parameters['k'] if trial.parameters['use_k'] else None,
        eps=trial.parameters['eps'] if not trial.parameters['use_k'] else None,
    )
elif PREPROCESSING == 'undirected_ppr':
    dataset = UndirectedPPRDataset(
        name=DATASET,
        use_lcc=USE_LCC,
        alpha=trial.parameters['alpha'],
        k=trial.parameters['k'] if trial.parameters['use_k'] else None,
        eps=trial.parameters['eps'] if not trial.parameters['use_k'] else None,
    )
elif PREPROCESSING == 'sdrfc':
    dataset = SDRFCDataset(
        name=DATASET,
        use_lcc=USE_LCC,
        max_steps=trial.parameters['max_steps'],
        remove_edges=trial.parameters['remove_edges'],
        tau=trial.parameters['tau'],
        is_undirected=False
    )
elif PREPROCESSING == 'sdrfcu':
    dataset = SDRFCDataset(
        name=DATASET,
        use_lcc=USE_LCC,
        max_steps=trial.parameters['max_steps'],
        remove_edges=trial.parameters['remove_edges'],
        tau=trial.parameters['tau'],
        is_undirected=True
    )
dataset.data = dataset.data.to(DEVICE)

model = GCN(
    dataset,
    hidden=trial.parameters['hidden_layers'] * [trial.parameters['hidden_units']],
    dropout=trial.parameters['dropout']
).to(DEVICE)

def train(model: torch.nn.Module, optimizer: Optimizer, data: Data):
    model.train()
    optimizer.zero_grad()
    logits = model(data)
    loss = F.nll_loss(logits[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

def evaluate(model: torch.nn.Module, data: Data, test: bool):
    model.eval()
    with torch.no_grad():
        logits = model(data)
    eval_dict = {}
    keys = ['val', 'test'] if test else ['val']
    for key in keys:
        mask = data[f'{key}_mask']
        # loss = F.nll_loss(logits[mask], data.y[mask]).item()
        # eval_dict[f'{key}_loss'] = loss
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        eval_dict[f'{key}_acc'] = acc
    return eval_dict


def run(dataset: InMemoryDataset,
        model: torch.nn.Module,
        seeds: np.ndarray,
        test: bool = False,
        max_epochs: int = 10000,
        patience: int = 100,
        lr: float = 0.01,
        weight_decay: float = 0.01,
        num_development: int = 1500,
        device: str = 'cuda'):
    start_time = time.perf_counter()

    best_dict = defaultdict(list)

    for seed in seeds:
        if num_development > 0:
            dataset.data = set_train_val_test_split(
                seed,
                dataset.data,
                num_development=num_development,
            ).to(device)
        else:
            dataset.data = set_train_val_test_split_webkb(
                seed,
                dataset.data,
                num_development=int(dataset.data.y.shape[0]*0.5),
                train_proportion=0.66,
            ).to(device)
        model.to(device).reset_parameters()
        optimizer = Adam(
            [
                {'params': model.non_reg_params, 'weight_decay': 0},
                {'params': model.reg_params, 'weight_decay': weight_decay}
            ],
            lr=lr
        )

        patience_counter = 0
        tmp_dict = {'val_acc': 0}

        for epoch in range(1, max_epochs + 1):
            if patience_counter == patience:
                break

            train(model, optimizer, dataset.data)
            eval_dict = evaluate(model, dataset.data, test)

            if eval_dict['val_acc'] < tmp_dict['val_acc']:
                patience_counter += 1
            else:
                patience_counter = 0
                tmp_dict['epoch'] = epoch
                for k, v in eval_dict.items():
                    tmp_dict[k] = v

        for k, v in tmp_dict.items():
            best_dict[k].append(v)

    client.send_metrics(trial=trial, iteration=epoch, objective=np.mean(best_dict['val_acc']))
            
    best_dict['duration'] = time.perf_counter() - start_time
    return dict(best_dict)

best_dict = run(
    dataset,
    model,
    seeds=test_seeds,
    lr=trial.parameters['lr'],
    weight_decay=trial.parameters['weight_decay'],
    test=True,
    num_development=trial.parameters['num_development'],
    device=DEVICE,
    patience=25
)

print(f"Here!{best_dict}")
