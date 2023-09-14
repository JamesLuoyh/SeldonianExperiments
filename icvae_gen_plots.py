## Run VFAE experiments as described in this example: 
### Possible constraint names: 
# [

# ]
### Possible epsilon values:
# 0.01 

import argparse
import numpy as np
import os
from experiments.generate_plots import SupervisedPlotGenerator
from experiments.base_example import BaseExample
from experiments.utils import probabilistic_accuracy, probabilistic_auc, demographic_parity
from seldonian.utils.io_utils import load_pickle
from sklearn.model_selection import train_test_split
from seldonian.dataset import SupervisedDataSet
import torch

ADULTS = "adults"
GERMAN = "german"

def vfae_example(
    spec_rootdir,
    results_base_dir,
    constraints = [],
    epsilons=[0.01, 0.05, 0.1],
    n_trials=10,
    data_fracs=np.logspace(-3,0,2),
    baselines = [],
    performance_metric="auc",
    n_workers=1,
    dataset=ADULTS,
):  
    data_fracs = [1]  #0.001,0.01,0.05,0.1,0.33,0.66,1
    # adult
    batch_epoch_dict = {
    #   0.001:[30,1000],
    #   0.005:[50,1000],
      0.01:[16,200],
      0.05:[20,200],
      0.1:[32,150],
    #   0.33:[32,300],
    #   0.66:[32,200],
      0.5:[32,150],
      1.0: [32,150]
    }
    # adult adversarial 5
    # batch_epoch_dict = {
    #   0.001:[10,2000],
    #   0.01:[20,1000],
    #   0.05:[50,600],
    #   0.1:[100,400],
    #   0.5:[100,400],
    #   1.0: [100,400]
    # }
    # debug
    # batch_epoch_dict = {
    #   0.001:[4,1],
    #   0.01:[16,1],
    #   0.05:[16,1],
    #   0.1:[16,1],
    #   0.33:[16,1],
    #   0.66:[16,1],
    #   1.0: [16,1]
    # }
    if performance_metric == "auc":
        perf_eval_fn = probabilistic_auc
    elif performance_metric == "accuracy":
        perf_eval_fn = probabilistic_accuracy
    elif performance_metric == "dp":
        perf_eval_fn = demographic_parity
    else:
        raise NotImplementedError(
            "Performance metric must be 'auc' or 'accuracy' or 'dp' for this example")
    for epsilon in epsilons:
        specfile = os.path.join(
            spec_rootdir,
            f"icvae_{dataset}_mutual_information_{epsilon}.pkl"
        )
        spec = load_pickle(specfile)
        results_dir = os.path.join(results_base_dir,
            f"icvae_{dataset}_mutual_information_{epsilon}_{performance_metric}")
        plot_savename = os.path.join(
            results_dir, f"icvae_{dataset}_mutual_information__{epsilon}_{performance_metric}.pdf"
        )

        orig_features = spec.dataset.features
        orig_labels = spec.dataset.labels
        orig_sensitive_attrs = spec.dataset.sensitive_attrs
        # First, shuffle features
        (train_features,test_features,train_labels,
        test_labels,train_sensitive_attrs,
        test_sensitive_attrs
            ) = train_test_split(
                orig_features,
                orig_labels,
                orig_sensitive_attrs,
                shuffle=True,
                test_size=0.2,
                random_state=42)
        new_dataset = SupervisedDataSet(
        features=train_features, 
        labels=train_labels,
        sensitive_attrs=train_sensitive_attrs, 
        num_datapoints=len(train_features),
        meta_information=spec.dataset.meta_information)
        # Set spec dataset to this new dataset
        spec.dataset = new_dataset
        # Setup performance evaluation function and kwargs 
        perf_eval_kwargs = {
            'X':test_features,
            'y':test_labels,
            'performance_metric':['auc', 'dp'],
            'device': torch.device(0),
            "eval_batch_size": 1000
        }
        constraint_eval_kwargs = {
            "eval_batch_size": 1000
        }

        plot_generator = SupervisedPlotGenerator(
            spec=spec,
            n_trials=n_trials,
            data_fracs=data_fracs,
            n_workers=n_workers,
            batch_epoch_dict=batch_epoch_dict,
            datagen_method='resample',
            perf_eval_fn=[probabilistic_auc, demographic_parity],
            constraint_eval_fns=[],
            results_dir=results_dir,
            perf_eval_kwargs=perf_eval_kwargs,
            constraint_eval_kwargs=constraint_eval_kwargs
        )
        plot_generator.run_seldonian_experiment(verbose=verbose)
        for baseline_model in baselines:
            plot_generator.run_baseline_experiment(
                model_name=baseline_model, verbose=verbose
            )
        plot_generator.make_plots(
            fontsize=12,
            legend_fontsize=8,
            performance_label=['auc', 'dp'],
            performance_yscale="linear",
            savename=plot_savename,
        )
        # ex = BaseExample(spec=spec)
        
        # ex.run(
        #     n_trials=n_trials,
        #     data_fracs=data_fracs,
        #     batch_epoch_dict=batch_epoch_dict,
        #     results_dir=results_dir,
        #     perf_eval_fn=perf_eval_fn,
        #     n_workers=n_workers,
        #     datagen_method="resample",
        #     verbose=False,
        #     baselines=baselines,
        #     performance_label=performance_metric,
        #     performance_yscale="linear",
        #     plot_savename=plot_savename,
        #     plot_fontsize=12,
        #     legend_fontsize=8,
        # )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Description of your program')
    # parser.add_argument('--constraint', help='Constraint to run', required=True)
    # parser.add_argument('--epsilon', help='Constraint to run', required=True)
    # parser.add_argument('--n_trials', help='Number of trials to run', required=True)
    # parser.add_argument('--n_workers', help='Number of workers to use', required=True)
    parser.add_argument('--include_baselines', help='include_baselines', action="store_true")
    parser.add_argument('--verbose', help='verbose', action="store_true")

    args = parser.parse_args()

    # constraint = args.constraint
    # epsilon = float(args.epsilon)
    # n_trials = int(args.n_trials)
    # n_workers = int(args.n_workers)
    include_baselines = args.include_baselines
    verbose = args.verbose

    if include_baselines:
        baselines = ["icvae_baseline"] # , 
    else:
        baselines = []

    performance_metric="dp"

    results_base_dir = f"/media/yuhongluo/SeldonianExperimentResults"
    dataset = ADULTS
    vfae_example(
        spec_rootdir="/media/yuhongluo/SeldonianExperimentSpecs/vfae/spec",
        results_base_dir=results_base_dir,
        # constraints = [constraint],
        # epsilons=[epsilon],
        # n_trials=n_trials,
        performance_metric=performance_metric,
        dataset = dataset,
        baselines = baselines  
    )
    
