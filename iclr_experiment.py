import argparse
from functools import partial
import numpy as np
import os
from ray import tune
from ray.tune import CLIReporter
from ray.tune.integration.wandb import WandbLogger
from ray.tune.logger import DEFAULT_LOGGERS
from ray.tune.schedulers import ASHAScheduler
import seaborn as sns
import torch

from gdl.data import BaseDataset, PPRDataset, SDRFDataset
from gdl.experiment.node_classification import evaluate, train
from gdl.experiment.optimizer import get_optimizer
from gdl.experiment.splits import set_train_val_test_split_robust as set_train_val_test_split
# from gdl.experiment.splits import set_train_val_test_split_classic as set_train_val_test_split
# from gdl.experiment.splits import set_train_val_test_split
from gdl.models import GCN
from gdl.seeds import test_seeds


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
    # opt["hidden_layers"] = tune.choice([1, 3])  # [1,3]
    # opt["hidden_units"] = tune.sample_from(lambda _: 2 ** np.random.randint(6, 9))
    # opt["dropout"] = 0.5
    # opt["lr"] = tune.loguniform(0.005, 0.03)
    # opt["weight_decay"] = tune.loguniform(0.001, 0.2)

    if opt["preprocessing"] == "none":
        pass
    elif opt["preprocessing"] == "ppr":
        opt["alpha"] = tune.loguniform(0.01, 0.12)
        opt["k"] = tune.choice([16, 32, 64])
        opt["eps"] = tune.loguniform(0.0001, 0.001)
        opt["use_k"] = tune.choice([True, False])
    elif opt["preprocessing"] == "sdrf":
        if opt["dataset"] in ["Cornell", "Wisconsin", "Texas"]:
            opt["tau"] = tune.loguniform(10, 200)
            opt["max_steps"] = tune.uniform(20, 500)
        else:
            opt["tau"] = tune.loguniform(30, 600)
            opt["max_steps"] = tune.uniform(200, 2000)
        opt["remove_edges"] = True
        # Folded normal distribution for removal_bound
        opt["removal_bound"] = tune.sample_from(
            lambda _: abs(np.random.normal(0.5, 4))
            if np.random.uniform(0, 1) < 0.85
            else 0
        )

    return opt


def get_test_results(models, datas):
    eval_dicts = [
        evaluate(model, data, test=True) for model, data in zip(models, datas)
    ]
    test_accs = [dict["test_acc"] for dict in eval_dicts]
    val_accs = [dict["val_acc"] for dict in eval_dicts]
    return test_accs, val_accs

def loguniform(low, high, size=None):
    return np.exp(np.random.uniform(np.log(low), np.log(high), size))

def train_ray(
    opt, checkpoint_dir=None, data_dir="../../digl/data", patience=25, test=True
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = get_preprocessed_dataset(opt, data_dir)
    # todo change seeds and num development and n_reps

    # models = []
    # datas = []
    # optimizers = []

    seeds = test_seeds[0 : opt["num_splits"]]
    setups_by_seed = {}

    model_configs = [
        {
            "hidden_layers": np.random.choice(range(1, opt["max_layers"] + 1)),
            "hidden_units": np.random.choice([16, 32, 64, 128]),
            "dropout": np.random.uniform(0.2, 0.8),
            "lr": loguniform(0.005,0.03),
            "weight_decay": loguniform(0.001, 0.2),
        }
        for _ in range(opt["models_per_seed"])
    ]

    for seed in seeds:
        dataset.data = set_train_val_test_split(
            seed,
            dataset.data,
            # num_development=1500,
            val_frac=0.2,
            test_frac=0.2
        ).to(device)
        # datas.append(dataset.data)

        setups_by_seed[seed] = {
            "data": dataset.data,
            "models": [],
            "optimizers": [],
        }

        for i in range(opt["models_per_seed"]):
            model_config = model_configs[i]
            model = GCN(
                dataset,
                hidden=model_config["hidden_layers"] * [model_config["hidden_units"]],
                dropout=model_config["dropout"],
            ).to(device)
            # model = GCN(
            #     dataset,
            #     hidden=opt["hidden_layers"] * [opt["hidden_units"]],
            #     dropout=opt["dropout"],
            # ).to(device)
            optimizer = get_optimizer(
                opt["optimizer"], model, lr=model_config["lr"], weight_decay=model_config["weight_decay"]
            )
            setups_by_seed[seed]["optimizers"].append(optimizer)
            setups_by_seed[seed]["models"].append(model)

            if checkpoint_dir:
                checkpoint = os.path.join(checkpoint_dir, "checkpoint")
                model_state, optimizer_state = torch.load(checkpoint)
                model.load_state_dict(model_state)
                optimizer.load_state_dict(optimizer_state)

    best_val_acc = 0
    best_test_acc = 0
    best_test_ci = 0
    best_config = {}

    for epoch in range(1, opt["epoch"] + 1):
        for i in range(opt["models_per_seed"]):
            loss = np.mean(
                [
                    train(model, optimizer, data)
                    for model, optimizer, data in zip(
                        [setups_by_seed[seed]["models"][i] for seed in seeds],
                        [setups_by_seed[seed]["optimizers"][i] for seed in seeds],
                        [setups_by_seed[seed]["data"] for seed in seeds],
                    )
                ]
            )
            test_accs, val_accs = get_test_results(
                [setups_by_seed[seed]["models"][i] for seed in seeds],
                [setups_by_seed[seed]["data"] for seed in seeds],
            )
            # print('Here!', test_accs, flush=True)

            # with tune.checkpoint_dir(step=epoch) as checkpoint_dir:
            #     best = np.argmax(val_accs)
            #     path = os.path.join(checkpoint_dir, "checkpoint")
            #     torch.save(([setups_by_seed[seed]['models'][i] for seed in seeds][best].state_dict(), [setups_by_seed[seed]['optimizers'][i] for seed in seeds][best].state_dict()), path)

            val_acc_mean = np.mean(val_accs)
            boots_series = sns.algorithms.bootstrap(
                test_accs, func=np.mean, n_boot=1000
            )
            test_acc_mean = np.mean(test_accs)
            test_acc_ci = np.max(np.abs(sns.utils.ci(boots_series, 95) - test_acc_mean))

            if val_acc_mean > best_val_acc:
                best_val_acc = val_acc_mean
                best_test_acc = test_acc_mean
                best_test_ci = test_acc_ci
                best_config = model_configs[i]

        tune.report(
            loss=loss,
            # val_acc=val_acc_mean,
            # test_acc=test_acc_mean,
            # test_acc_ci=test_acc_ci,
            best_val_acc=best_val_acc,
            best_test_acc=best_test_acc,
            best_test_ci=best_test_ci,
            hidden_layers=best_config["hidden_layers"],
            hidden_units=best_config["hidden_units"],
            dropout=best_config["dropout"],
        )


def main(opt):
    print(f'Data directory: {opt["data_dir"]}')
    opt["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    datasets = opt["dataset"]
    preprocessings = opt["preprocessing"]
    original_undirected_flag = opt["undirected"]
    for dataset in datasets:
        for preprocessing in preprocessings:
            opt["dataset"] = dataset
            if preprocessing[-1] == "u":
                opt["preprocessing"] = preprocessing[:-1]
                opt["undirected"] = True
            else:
                opt["preprocessing"] = preprocessing
                opt["undirected"] = original_undirected_flag
            if opt["use_wandb"]:
                opt["wandb"] = {
                    "project": f'{opt["wandb_project"]} ({dataset})',
                    "log_config": True,
                    "group": f"{dataset}_{opt['preprocessing']}{'_undirected' if opt['undirected'] else ''}",
                }
            opt = set_search_space(opt)

            # todo remove after debugging
            scheduler = ASHAScheduler(
                metric="best_val_acc",
                mode="max",
                max_t=opt["epoch"],
                grace_period=opt["grace_period"],
                reduction_factor=opt["reduction_factor"],
            )
            reporter = CLIReporter(metric_columns=["accuracy", "test_acc", "conf_int"])
            # choose a search algorithm from https://docs.ray.io/en/latest/tune/api_docs/suggestion.html
            search_alg = None
            # todo this won't work as preprocessing is a tune.choice object
            # experiment_name = opt['dataset'][:4] + '_' + opt['preprocessing']

            tune.run(
                partial(train_ray, data_dir=opt["data_dir"]),
                name=opt["name"],
                resources_per_trial={"cpu": opt["cpus"], "gpu": opt["gpus"]},
                search_alg=search_alg,
                keep_checkpoints_num=3,
                checkpoint_score_attr=opt["metric"],
                config=opt,
                num_samples=opt["num_samples"],
                scheduler=scheduler,
                max_failures=2,
                local_dir="ray_tune",
                progress_reporter=reporter,
                raise_on_failed_trial=False,
                loggers=DEFAULT_LOGGERS + (WandbLogger,),
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimizer", type=str, default="adam", help="Optimizer.")
    parser.add_argument(
        "--dataset",
        nargs="+",
        type=str,
        default="Cora",
        help="Cora, Citeseer, Pubmed, Computers, Photo, CoauthorCS",
    )
    parser.add_argument("--preprocessing", nargs="+", type=str, default="none")
    parser.add_argument(
        "--epoch",
        type=int,
        default=100,
        help="Number of training epochs per iteration.",
    )
    # ray args
    parser.add_argument(
        "--num_samples", type=int, default=32, help="number of ray trials"
    )
    parser.add_argument(
        "--gpus",
        type=float,
        default=1,
        help="Number of gpus per trial. Can be fractional",
    )
    parser.add_argument(
        "--cpus",
        type=float,
        default=2,
        help="Number of cpus per trial. Can be fractional",
    )
    parser.add_argument(
        "--grace_period",
        type=int,
        default=5,
        help="Number of epochs to wait before terminating trials",
    )
    parser.add_argument(
        "--reduction_factor",
        type=int,
        default=4,
        help="Number of trials is halved after this many epochs",
    )
    parser.add_argument("--name", type=str, default="ray_exp")
    parser.add_argument(
        "--num_splits",
        type=int,
        default=5,
        help="Number of random splits >= 0. 0 for planetoid split",
    )
    parser.add_argument(
        "--num_init", type=int, default=1, help="Number of random initializations >= 0"
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="accuracy",
        help="Metric to sort the hyperparameter tuning runs on",
    )
    parser.add_argument("--use_lcc", dest="use_lcc", action="store_true")
    parser.add_argument(
        "--use_wandb", dest="use_wandb", action="store_true", default=False
    )
    parser.add_argument("--wandb_project")
    parser.add_argument("--wandb_group")
    parser.add_argument("--not_lcc", dest="use_lcc", action="store_false")
    parser.add_argument("--data_dir", dest="data_dir", default="~/data")
    parser.add_argument("--undirected", action="store_true", default=False)
    parser.add_argument("--models_per_seed", type=int)
    parser.add_argument("--max_layers", type=int, default=3)
    parser.set_defaults(use_lcc=True)
    args = parser.parse_args()
    opt = vars(args)
    main(opt)
