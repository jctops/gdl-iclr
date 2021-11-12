import numpy as np
from ray import tune

from gdl.data import BaseDataset, PPRDataset, SDRFDataset
from gdl.experiment.node_classification import evaluate, train
from gdl.models import GCN


def get_preprocessed_dataset(opt, data_dir):
    if opt["preprocessing"] == "none":
        dataset = BaseDataset(
            name=opt["dataset"],
            use_lcc=opt["use_lcc"],
            undirected=opt["undirected"],
            data_dir=data_dir,
        )
    elif opt["preprocessing"] == "ppr":
        dataset = PPRDataset(
            name=opt["dataset"],
            use_lcc=opt["use_lcc"],
            alpha=opt["alpha"],
            k=opt["k"] if opt["use_k"] else None,
            eps=opt["eps"] if not opt["use_k"] else None,
            undirected=opt["undirected"],
            data_dir=data_dir,
        )
    elif opt["preprocessing"] == "sdrf":
        dataset = SDRFDataset(
            name=opt["dataset"],
            use_lcc=opt["use_lcc"],
            max_steps=opt["max_steps"],
            remove_edges=opt["remove_edges"],
            removal_bound=opt["removal_bound"],
            tau=opt["tau"],
            undirected=opt["undirected"],
            data_dir=data_dir,
        )
    dataset.data = dataset.data.to(opt["device"])
    return dataset


def set_search_space(opt):
    opt["num_development"] = 1500
    opt["hidden_layers"] = tune.choice([1, 2])  # [1,3]
    opt["hidden_units"] = tune.sample_from(lambda _: 2 ** np.random.randint(6, 9))
    opt["dropout"] = 0.5
    opt["lr"] = tune.loguniform(0.005, 0.03)
    opt["weight_decay"] = tune.loguniform(0.0001, 0.05)

    if opt["preprocessing"] == "none":
        pass
    elif opt["preprocessing"] == "ppr":
        opt["alpha"] = tune.loguniform(0.01, 0.12)
        opt["k"] = tune.choice([16, 32, 64])
        opt["eps"] = tune.loguniform(0.0001, 0.001)
        opt["use_k"] = tune.choice([True, False])
    elif opt["preprocessing"] == "sdrf":
        opt["tau"] = tune.loguniform(250, 600)
        opt["max_steps"] = tune.choice([500, 1000])
        opt["remove_edges"] = True
        # Folded normal distribution for removal_bound
        opt["removal_bound"] = tune.sample_from(lambda _: abs(np.random.normal(0.5, 4)))

    return opt


def get_test_results(models, datas):
    eval_dicts = [
        evaluate(model, data, test=True) for model, data in zip(models, datas)
    ]
    test_accs = [dict["test_acc"] for dict in eval_dicts]
    val_accs = [dict["val_acc"] for dict in eval_dicts]
    return test_accs, val_accs


def train_ray():
    pass

