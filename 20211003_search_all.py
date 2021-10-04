import ast
from multiprocessing import Pool
import numpy as np
import pickle as pkl
import seaborn as sns
import sherpa
import time

def get_parameters(dataset, preprocessing, output_dir, **kwargs):
    parameters = [
        sherpa.Choice(name='dataset', range=[dataset]),
        sherpa.Choice(name='preprocessing', range=[(preprocessing if not preprocessing.startswith('sdrfc') else 'sdrfc') if not preprocessing.startswith('sdrfcu') else 'sdrfcu']),
        sherpa.Choice(name='output_dir', range=[output_dir]),
        sherpa.Choice(name='num_development', range=[kwargs['num_development']]),
        sherpa.Choice(name='device', range=['cuda']),
        sherpa.Discrete(name='hidden_layers', range=[1,2]), # [1,3]
        sherpa.Ordinal(name='hidden_units', range=[16,32,64,128]),
        sherpa.Continous(name='dropout', range=[0.2,0.8]),
        sherpa.Continous(name='lr', range=[0.005,0.05], scale='log'),
        sherpa.Continous(name='weight_decay', range=[0.01,0.2], scale='log'),
    ]
    if (preprocessing == 'none') or (preprocessing == 'undirected'):
        pass
    elif (preprocessing == 'ppr') or (preprocessing == 'undirected_ppr'):
        parameters += [
            sherpa.Continuous(name='alpha', range=[0.01, 0.12], scale='log'),
            sherpa.Ordinal(name='k', range=[16,32,64]),
            sherpa.Continous(name='epsilon', range=[0.0001, 0.001], scale='log'),
            sherpa.Choice(name='use_k', range=[True, False]),
        ]
    elif (preprocessing == 'sdrfct') or (preprocessing == 'sdrfcut'):
        parameters += [
            sherpa.Discrete(name='max_steps', range=kwargs['max_steps_range']),
            sherpa.Choice(name='remove_edges', range=[True]),
            sherpa.Discrete(name='tau', range=kwargs['tau_range'], scale='log'),
        ]
    elif (preprocessing == 'sdrfcf') or (preprocessing == 'sdrfcuf'):
        parameters += [
            sherpa.Discrete(name='max_steps', range=kwargs['max_steps_range']),
            sherpa.Choice(name='remove_edges', range=[False]),
            sherpa.Discrete(name='tau', range=kwargs['tau_range'], scale='log'),
        ]
    return parameters


class Experiment:
    def __init__(self, dataset, preprocessing, search_script, max_num_trials=50, sleep_t=0, **kwargs):
        self.dataset = dataset
        self.preprocessing = preprocessing
        self.search_script = search_script
        self.max_num_trials = max_num_trials
        self.sleep_t = sleep_t
        self.kwargs = kwargs

def gen_experiments():
    print('generating experiments')
    experiments = []
    t = 0

    # WebKB experiments
    # for dataset in ['Cornell', 'Wisconsin', 'Texas']:
    #     for preprocessing in ['none', 'undirected', 'ppr', 'undirected_ppr']:
    #         experiments.append(
    #             Experiment(
    #                 dataset,
    #                 preprocessing,
    #                 '20211003_search_experiment.py',
    #                 num_development=0,
    #                 hours=1,
    #                 max_num_trials=40,
    #                 sleep_t=t,
    #             )
    #         )
    #         t += 15
    #     for preprocessing in ['sdrfct', 'sdrfcf', 'sdrfcut', 'sdrfcuf']:
    #         experiments.append(
    #             Experiment(
    #                 dataset,
    #                 preprocessing,
    #                 '20211003_search_experiment.py',
    #                 tau_range=[50,200],
    #                 max_steps_range=[10,1000],
    #                 num_development=0,
    #                 hours=1,
    #                 max_num_trials=100,
    #                 sleep_t=t,
    #             )
    #         )
    #         t += 15

    # DIGL experiments #1
    for dataset in ['Cora', 'Citeseer']:
        for preprocessing in ['none', 'undirected', 'ppr', 'undirected_ppr']:
            experiments.append(
                Experiment(
                    dataset,
                    preprocessing,
                    '20211003_search_experiment.py',
                    num_development=1500,
                    hours=1,
                    max_num_trials=40,
                    sleep_t=t,
                )
            )
            t += 15
        for preprocessing in ['sdrfct', 'sdrfcf', 'sdrfcut', 'sdrfcuf']:
            experiments.append(
                Experiment(
                    dataset,
                    preprocessing,
                    '20211003_search_experiment.py',
                    tau_range=[50,400],
                    max_steps_range=[300,3000],
                    num_development=1500,
                    hours=2,
                    max_num_trials=100,
                    sleep_t=t,
                )
            )
            t += 15

    # DIGL experiments #2
    for dataset in ['Pubmed']:
        for preprocessing in ['none', 'undirected', 'ppr', 'undirected_ppr']:
            experiments.append(
                Experiment(
                    dataset,
                    preprocessing,
                    '20211003_search_experiment.py',
                    num_development=1500,
                    hours=1,
                    max_num_trials=40,
                    sleep_t=t,
                )
            )
            t += 15
        for preprocessing in ['sdrfct', 'sdrfcf', 'sdrfcut', 'sdrfcuf']:
            experiments.append(
                Experiment(
                    dataset,
                    preprocessing,
                    '20211003_search_experiment.py',
                    tau_range=[50,400],
                    max_steps_range=[300,3000],
                    num_development=1500,
                    hours=3,
                    max_num_trials=100,
                    sleep_t=t,
                )
            )
            t += 15

    # Wikipedia experiments #1
    for dataset in ['Chameleon']:
        for preprocessing in ['none', 'undirected', 'ppr', 'undirected_ppr']:
            experiments.append(
                Experiment(
                    dataset,
                    preprocessing,
                    '20211003_search_experiment.py',
                    num_development=600,
                    hours=1,
                    max_num_trials=40,
                    sleep_t=t,
                )
            )
            t += 15
        for preprocessing in ['sdrfct', 'sdrfcf', 'sdrfcut', 'sdrfcuf']:
            experiments.append(
                Experiment(
                    dataset,
                    preprocessing,
                    '20211003_search_experiment.py',
                    tau_range=[30,400],
                    max_steps_range=[200,2500],
                    num_development=600,
                    hours=2,
                    max_num_trials=100,
                    sleep_t=t,
                )
            )
            t += 15

    # Wikipedia experiments #2
    for dataset in ['Squirrel', 'Actor']:
        for preprocessing in ['none', 'undirected', 'ppr', 'undirected_ppr']:
            experiments.append(
                Experiment(
                    dataset,
                    preprocessing,
                    '20211003_search_experiment.py',
                    num_development=1500,
                    hours=1,
                    max_num_trials=40,
                    sleep_t=t,
                )
            )
            t += 15
        for preprocessing in ['sdrfct', 'sdrfcf', 'sdrfcut', 'sdrfcuf']:
            experiments.append(
                Experiment(
                    dataset,
                    preprocessing,
                    '20211003_search_experiment.py',
                    tau_range=[50,400],
                    max_steps_range=[200,3000],
                    num_development=1500,
                    hours=3,
                    max_num_trials=100,
                    sleep_t=t,
                )
            )
            t += 15
    print('generated {len(experiments)} experiments')

    return experiments

experiments = gen_experiments()


def run_experiment(experiment, max_num_trials=100, max_concurrent=None):
    time.sleep(experiment.sleep_t)
    experiment_name = experiment.dataset[:4]+'_'+experiment.preprocessing
    print(f'running {experiment_name}')
    max_num_trials = experiment.max_num_trials
    if max_concurrent is None:
        max_concurrent = max_num_trials
    output_dir = './sherpa_final/'+experiment_name
    parameters = get_parameters(experiment.dataset, experiment.preprocessing, output_dir, **experiment.kwargs)
    algorithm = sherpa.algorithms.RandomSearch(max_num_trials=max_num_trials)
    # scheduler = sherpa.schedulers.SGEScheduler(submit_options=f'-N {experiment_name} -l h_rt={experiment.kwargs["hours"]}:0:0,h_vmem=8G,estmem=8G,gpu_slots=1 -j y -S /bin/bash',
    #     environment='/home/jtopping/.julia/conda/3/bin/activate && conda activate env_clr && module load cuda-10.1.243')
    # todo need to make this a local scheduler
    # so = f'-N {experiment_name} -l h_rt={experiment.kwargs["hours"]}:0:0,h_vmem=8G,estmem=8G,gpu_slots=1 -j y -S /bin/bash'
    scheduler = sherpa.schedulers.LocalSchedular()

    results = sherpa.optimize(
        parameters=parameters,
        algorithm=algorithm,
        lower_is_better=False,
        filename=f'./{experiment.search_script}',
        scheduler=scheduler,
        verbose=0,
        max_concurrent=max_concurrent,
        output_dir=output_dir,
        disable_dashboard=True
    )

    best_trial_id = results['Trial-ID']
    with open(f'{output_dir}/jobs/trial_{best_trial_id}.out', 'r+') as f:
        lines = f.readlines()
        for l in lines[::-1]:
            if l.startswith('Here!'):
                test_accs = ast.literal_eval(l[5:])['test_acc']
                break
        boots_series = sns.algorithms.bootstrap(test_accs, func=np.mean, n_boot=1000)
        test_acc_mean = np.mean(test_accs)
        test_acc_ci = np.max(np.abs(sns.utils.ci(boots_series, 95) - test_acc_mean))

    pkl.dump(
        (experiment_name, test_acc_mean, test_acc_ci, results),
        open(
            output_dir+'/best_trial.pkl',
            'wb'
        )
    )

    return experiment_name, test_acc_mean, test_acc_ci

with Pool(len(experiments)) as p:
    results = p.map(run_experiment, experiments)
pkl.dump(
    results,
    open(
        './results_1.pkl',
        'wb'
    )
)
