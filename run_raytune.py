from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

import ast
from multiprocessing import Pool
import numpy as np
import pickle as pkl
import seaborn as sns
import torch
import torch.nn.functional as F
from torch.optim import Adam, Optimizer
from torch_geometric.data import Data, InMemoryDataset
import argparse
from digl.seeds import val_seeds, test_seeds


from functools import partial
import os
from collections import defaultdict

from digl.models import GCN
from digl.data import get_dataset, PPRDataset, SDRFCDataset, UndirectedDataset, UndirectedPPRDataset, \
  set_train_val_test_split, set_train_val_test_split_webkb
from digl.seeds import val_seeds, test_seeds


def train(model: torch.nn.Module, optimizer: Optimizer, data: Data):
  model.train()
  optimizer.zero_grad()
  logits = model(data)
  loss = F.nll_loss(logits[data.train_mask], data.y[data.train_mask])
  loss.backward()
  optimizer.step()
  return loss.item()


def evaluate(model: torch.nn.Module, data: Data, test: bool):
  model.eval()
  with torch.no_grad():
    logits = model(data)
  eval_dict = {}
  keys = ['val', 'test'] if test else ['val']
  for key in keys:
    mask = data[f'{key}_mask']
    pred = logits[mask].max(1)[1]
    acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
    eval_dict[f'{key}_acc'] = acc
  return eval_dict


def get_optimizer(name, parameters, lr, weight_decay=0):
  if name == 'sgd':
    return torch.optim.SGD(parameters, lr=lr, weight_decay=weight_decay)
  elif name == 'rmsprop':
    return torch.optim.RMSprop(parameters, lr=lr, weight_decay=weight_decay)
  elif name == 'adagrad':
    return torch.optim.Adagrad(parameters, lr=lr, weight_decay=weight_decay)
  elif name == 'adam':
    return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
  elif name == 'adamax':
    return torch.optim.Adamax(parameters, lr=lr, weight_decay=weight_decay)
  else:
    raise Exception("Unsupported optimizer: {}".format(name))


def get_preprocessed_dataset(opt, data_dir):
  if opt['preprocessing'] == 'none':
    dataset = get_dataset(name=opt['dataset'], use_lcc=opt['use_lcc'], data_dir=data_dir)
  elif opt['preprocessing'] == 'undirected':
    dataset = UndirectedDataset(name=opt['dataset'], use_lcc=opt['use_lcc'], )
  elif opt['preprocessing'] == 'ppr':
    dataset = PPRDataset(name=opt['dataset'], use_lcc=opt['use_lcc'], alpha=opt['alpha'],
                         k=opt['k'] if opt['use_k'] else None,
                         eps=opt['eps'] if not opt['use_k'] else None,
                         )
  elif opt['preprocessing'] == 'undirected_ppr':
    dataset = UndirectedPPRDataset(name=opt['dataset'], use_lcc=opt['use_lcc'], alpha=opt['alpha'],
                                   k=opt['k'] if opt['use_k'] else None,
                                   eps=opt['eps'] if not opt['use_k'] else None,
                                   )
  elif opt['preprocessing'] in ['sdrfct', 'sdrfcf']:
    dataset = SDRFCDataset(name=opt['dataset'], use_lcc=opt['use_lcc'], max_steps=opt['max_steps'],
                           remove_edges=opt['remove_edges'],
                           tau=opt['tau'], is_undirected=False, data_dir=data_dir)
  elif opt['preprocessing'] in ['sdrfcut', 'sdrfcuf']:
    dataset = SDRFCDataset(
      name=opt['dataset'],
      use_lcc=opt['use_lcc'],
      max_steps=opt['max_steps'],
      remove_edges=opt['remove_edges'],
      tau=opt['tau'],
      is_undirected=True, data_dir=data_dir
    )
  dataset.data = dataset.data.to(opt['device'])
  return dataset


def set_search_space(opt):
  opt['num_development'] = 1500
  opt['hidden_layers'] = tune.choice([1, 2])  # [1,3]
  opt['hidden_units'] = tune.sample_from(lambda _: 2 ** np.random.randint(6, 9))
  opt['dropout'] = 0.5
  opt['lr'] = tune.loguniform(0.005, 0.03)
  opt['weight_decay'] = tune.loguniform(0.0001, 0.05)
  if opt['preprocessing'] in ['none', 'undirected']:
    pass
  elif opt['preprocessing'] in ['ppr', 'undirected_ppr']:
    opt['alpha'] = tune.loguniform(0.01, 0.12)
    opt['k'] = tune.choice([16, 32, 64])
    opt['eps'] = tune.loguniform(0.0001, 0.001)
    opt['use_k'] = tune.choice([True, False])

  elif opt['preprocessing'] in ['sdrfct', 'sdrfcf', 'sdrfcut', 'sdrfcuf']:
    opt['tau'] = tune.loguniform(250, 600)
    opt['max_steps'] = tune.choice([500, 1000])
    opt['remove_edges'] = True

  if opt['preprocessing'] in ['sdrfcf', 'sdrfcuf']:
    opt['remove_edges'] = False
  return opt


def get_test_results(models, datas):
  eval_dicts = [evaluate(model, data, test=True) for model, data in zip(models, datas)]
  test_accs = [dict['test_acc'] for dict in eval_dicts]
  val_accs = [dict['val_acc'] for dict in eval_dicts]
  return test_accs, val_accs

def train_ray(opt, checkpoint_dir=None, data_dir="../../digl/data", patience=25, test=True):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  dataset = get_preprocessed_dataset(opt, data_dir)
  #todo change seeds and num development and n_reps

  models = []
  datas = []
  optimizers = []
  best_dict = defaultdict(list)
  for seed in enumerate(test_seeds[0:opt['num_splits']]):
    dataset.data = set_train_val_test_split(seed, dataset.data, num_development=1500,).to(device)
    model = GCN(dataset, hidden=opt['hidden_layers'] * [opt['hidden_units']], dropout=opt['dropout']).to(device)
    optimizer = Adam(
      [
        {'params': model.non_reg_params, 'weight_decay': 0},
        {'params': model.reg_params, 'weight_decay': opt['weight_decay']}
      ], lr=opt['lr'])
    models.append(model)
    datas.append(dataset.data)
    optimizers.append(optimizer)
    if checkpoint_dir:
      checkpoint = os.path.join(checkpoint_dir, "checkpoint")
      model_state, optimizer_state = torch.load(checkpoint)
      model.load_state_dict(model_state)
      optimizer.load_state_dict(optimizer_state)
    # patience_counter = 0
    #
    # tmp_dict = {'val_acc': 0}
    # model.to(device).reset_parameters()


  for epoch in range(1, opt['epoch'] + 1):
    # if patience_counter == patience:
    #   break

    # loss = train(model, optimizer, dataset.data)
    loss = np.mean([train(model, optimizer, data) for model, optimizer, data in zip(models, optimizers, datas)])
    # eval_dict = evaluate(model, dataset.data, test)
    test_accs, val_accs = get_test_results(models, datas)

    with tune.checkpoint_dir(step=epoch) as checkpoint_dir:
      best = np.argmax(val_accs)
      path = os.path.join(checkpoint_dir, "checkpoint")
      torch.save((models[best].state_dict(), optimizers[best].state_dict()), path)

    # test_accs = best_dict['test_acc']
    boots_series = sns.algorithms.bootstrap(test_accs, func=np.mean, n_boot=1000)
    test_acc_mean = np.mean(test_accs)
    test_acc_ci = np.max(np.abs(sns.utils.ci(boots_series, 95) - test_acc_mean))
    tune.report(loss=loss, accuracy=np.mean(val_accs), test_acc=test_acc_mean,
                conf_int=test_acc_ci)

  #   if eval_dict['val_acc'] < tmp_dict['val_acc']:
  #     patience_counter += 1
  #   else:
  #     patience_counter = 0
  #     tmp_dict['epoch'] = epoch
  #     for k, v in eval_dict.items():
  #       tmp_dict[k] = v
  #
  #   for k, v in tmp_dict.items():
  #     best_dict[k].append(v)
  #
  # print(f'best_dict: {best_dict}')
  # test_accs = best_dict['test_acc']
  # boots_series = sns.algorithms.bootstrap(test_accs, func=np.mean, n_boot=1000)
  # test_acc_mean = np.mean(test_accs)
  # test_acc_ci = np.max(np.abs(sns.utils.ci(boots_series, 95) - test_acc_mean))
  # tune.report(loss=loss, accuracy=np.mean(best_dict['val_acc']), test_acc=np.mean(best_dict['test_acc']), conf_int=test_acc_ci)


def main(opt):
  data_dir = os.path.abspath("digl/data")
  print(f'data directory: {data_dir}')
  opt['device'] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  # todo replace
  # for method in ['sdrfct', 'sdrfcf', 'sdrfcut', 'sdrfcuf']:
  for method in ['sdrfcuf']:
  # for method in ['sdrfct']:
    opt['preprocessing'] = method
    opt = set_search_space(opt)
    # todo remove after debugging
    scheduler = ASHAScheduler(
      metric='accuracy',
      mode="max",
      max_t=opt["epoch"],
      grace_period=opt["grace_period"],
      reduction_factor=opt["reduction_factor"],
    )
    reporter = CLIReporter(
      metric_columns=["accuracy", "test_acc", "conf_int"]
    )
    # choose a search algorithm from https://docs.ray.io/en/latest/tune/api_docs/suggestion.html
    search_alg = None
    # todo this won't work as preprocessing is a tune.choice object
    # experiment_name = opt['dataset'][:4] + '_' + opt['preprocessing']

    result = tune.run(
      partial(train_ray, data_dir=data_dir),
      name=opt['name'],
      resources_per_trial={"cpu": opt["cpus"], "gpu": opt["gpus"]},
      search_alg=search_alg,
      keep_checkpoints_num=3,
      checkpoint_score_attr=opt['metric'],
      config=opt,
      num_samples=opt["num_samples"],
      scheduler=scheduler,
      max_failures=2,
      local_dir="ray_tune",
      progress_reporter=reporter,
      raise_on_failed_trial=False,
    )


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--optimizer", type=str, default="adam", help="Optimizer.")
  parser.add_argument(
    "--dataset", type=str, default="Cora", help="Cora, Citeseer, Pubmed, Computers, Photo, CoauthorCS"
  )
  parser.add_argument("--epoch", type=int, default=100, help="Number of training epochs per iteration.")
  # ray args
  parser.add_argument("--num_samples", type=int, default=32, help="number of ray trials")
  parser.add_argument("--gpus", type=float, default=1, help="number of gpus per trial. Can be fractional")
  parser.add_argument("--cpus", type=float, default=2, help="number of cpus per trial. Can be fractional")
  parser.add_argument(
    "--grace_period", type=int, default=5, help="number of epochs to wait before terminating trials"
  )
  parser.add_argument(
    "--reduction_factor", type=int, default=4, help="number of trials is halved after this many epochs"
  )
  parser.add_argument("--name", type=str, default="ray_exp")
  parser.add_argument("--num_splits", type=int, default=5, help="Number of random splits >= 0. 0 for planetoid split")
  parser.add_argument("--num_init", type=int, default=1, help="Number of random initializations >= 0")
  parser.add_argument('--metric', type=str, default='accuracy',
                      help='metric to sort the hyperparameter tuning runs on')
  parser.add_argument('--use_lcc', dest='use_lcc', action='store_true')
  parser.add_argument('--not_lcc', dest='use_lcc', action='store_false')
  parser.set_defaults(use_lcc=True)
  args = parser.parse_args()
  opt = vars(args)
  main(opt)
