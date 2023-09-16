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
from experiments.utils import probabilistic_accuracy, probabilistic_auc, demographic_parity, multiclass_demographic_parity
from seldonian.utils.io_utils import load_pickle
from sklearn.model_selection import train_test_split
from seldonian.dataset import SupervisedDataSet
import torch
from seldonian.models.pytorch_vae import PytorchVFAE

ADULTS = "adults"
GERMAN = "german"
HEALTH = "health"
torch.manual_seed(2023)
np.random.seed(2023)
def vfae_example(
    spec_rootdir,
    results_base_dir,
    constraints = [],
    epsilons=[0.32],#[0.32],#0.32],#0.32],#[0.3],#[0.1],#[0.0069],0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1], #[0.0069],# 0.01, , 0.02],
    n_trials=10,#10,
    data_fracs=np.logspace(-3,0,5),
    baselines = [],
    performance_metric="auc",
    n_workers=1,
    dataset=ADULTS,
    validation=True,
    device_id=0,
):  
    data_fracs = [0.1,0.15,0.25,0.40,0.65, 1]#0.1,0.15,0.25,0.40,0.65, 1]#[1]#[0.1, 0.15,0.25,0.40,0.65, 1]# 0.15,0.25,0.40,0.65, 1][1, 0.65, 0.40, 0.25, 0.15, 0.1] #[1]#[0.1, 0.15,0.25,0.40,0.65, 1]#0.1, 0.15,0.25,0.40,0.65, [0.01, 0.025, , 0.15, 0.5, 1]  # 0.01, 0.025, 0.06, 0.15, 0.5, #0.001,0.01,0.05,0.1,0.33,0.66,1
    
    # for baseline



    ##############
    # Supervised for both baseline and seldonian
    # 
    ##############
    # batch_epoch_dict = {
    # #   0.001:[30,1000],
    # #   0.005:[50,1000],
    # #   0.01:[16,400],
    # #   0.025:[32, 400],
    # #   0.05:[32,400],
    # #   0.05:[32,400],
    # #   0.06:[400,500],
    #   0.1:[64,150],
    #   0.15:[64,150],
    #   0.25:[64, 150],
    #   0.40:[64, 50], #150 (1e-4), #50 (1e-3)
    #   0.65:[64, 50], #150 (1e-4), #50 (1e-3)
    # #   0.33:[32,300],
    # #   0.66:[32,200],
    #   0.5:[64,50], #150
    #   1.0: [64,50], #100 (1e-4), #50 (1e-3)
    # }
###############################################################################

    # best for seldonian adult
    # batch_epoch_dict = {
    # #   0.001:[30,1000],
    # #   0.005:[50,1000],
    # #   0.01:[16,400],
    # #   0.025:[32, 400],
    # #   0.05:[32,400],
    # #   0.05:[32,400],
    # #   0.06:[400,500],
    #   0.1:[64,150],
    #   0.15:[64,150],
    #   0.25:[64, 150],
    #   0.40:[64, 150],
    #   0.65:[64, 150],
    # #   0.33:[32,300],
    # #   0.66:[32,200],
    #   0.5:[64,150],
    #   1.0: [64,100]
    # }
    # adult adversarial 5
    # batch_epoch_dict = {
    #   0.001:[10,2000],
    #   0.01:[20,1000],
    #   0.05:[50,600],
    #   0.1:[100,400],
    #   0.5:[100,400],
    #   1.0: [100,400]
    # }
    # DEBUG
    batch_epoch_dict = {
      0.001:[500,1],
      
      0.1:[500,200],
      0.15: [500,134],
      0.25:[500,80],
      0.40:[500,50],
      0.65:[500,31],
      1.0: [500,40]
    } 

    batch_epoch_dict = {
        0.1:[500,40],
        0.15: [500,40],
        0.25:[500,40],
        0.40:[500,40],
        0.65:[500,40],
        1.0: [500,200],
    }

    # # Theorectical FRG
    # batch_epoch_dict = {      
    #   0.1:[500,200],
    #   0.15: [500,134],
    #   0.25:[500,80],
    #   0.40:[500,50],
    #   0.65:[500,31],
    #   1.0: [500,20]
    # }
    z_dim = 50
    device = torch.device(device_id)
    model = PytorchVFAE(device, **{"x_dim": 117,
        "s_dim": 1,
        "y_dim": 1,
        "z1_enc_dim": z_dim,
        "z2_enc_dim": z_dim,
        "z1_dec_dim": z_dim,
        "x_dec_dim": z_dim,
        "z_dim": z_dim,
        "dropout_rate": 0.0,
        "alpha_adv": 1e-3,
        "mi_version": 1}
    )

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
            f"vfae_unsupervised_{dataset}_mutual_information_{epsilon}.pkl"
        )
        spec = load_pickle(specfile)


        alpha_l = [1e-4]#[1e-4, 1e-3] #[1e-4, 1e-5]# [1e-4, 1e-5] #[1e-3, 1e-4]
        alpha_lambda_l = [1e-4]#[1e-3]#[1e-3,1e-4]#[1e-3, 1e-4] #[1e-2] #[1e-3, 1e-4]
        lambda_init_l = [0.2]#[0.05,0.1,0.2]#[.01, 0.1, 0.5, 1.0]#[1e-2]#[1e-1, 1e-2]#[0.1, 0.15,0.25,0.40,0.65, 1]
        epochs_l = [60]#30]#[60,90]#   , 90]#[150]#, 120, 150]#[175]# [200, 250]#, 100, 125, 150]#50, 75] #, 100#, 125, 150]
        delta_l = [0.1]#[0.5,0.7,0.9]#, 0.7, 0.9]

        for lambda_init in lambda_init_l:
            for alpha in alpha_l:
                for alpha_lambda in alpha_lambda_l:
                    for epochs in epochs_l:
                        for delta in delta_l:

                            spec.optimization_hyperparams["lambda_init"] = np.array([lambda_init])
                            spec.optimization_hyperparams["alpha_theta"] = alpha
                            spec.optimization_hyperparams["alpha_lamb"] = alpha_lambda
                            spec.parse_trees[0].deltas = [delta] 
                            batch_epoch_dict = {
                                0.1:[500,int(epochs/0.1)],
                                0.15: [500,int(epochs/0.15)],
                                0.25:[500,int(epochs/0.25)],
                                0.40:[500,int(epochs/0.40)],
                                0.65:[500,int(epochs/0.65)],
                                1.0: [500,epochs],
                            }

                            if validation:
                                suffix = "validation"
                            else:
                                suffix = "test"
                            results_dir = os.path.join(results_base_dir,
                                f"separate_tuning_{dataset}_mutual_information_{epsilon}_{alpha}_{alpha_lambda}_{lambda_init}_{epochs}_{delta}_{suffix}")
                            plot_savename = os.path.join(
                                results_dir, f"separate_tuning_{dataset}_mutual_information__{epsilon}_{alpha}_{alpha_lambda}_{lambda_init}_{epochs}_{delta}_{suffix}.pdf"
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
                                    random_state=2023)
                            # print(train_features.shape)
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
                                's_dim': orig_sensitive_attrs.shape[1]
                            }
                            # constraint_eval_kwargs = {
                            #     'eval_batch_size':500
                            # }
                            plot_generator = SupervisedPlotGenerator(
                                spec=spec,
                                n_trials=n_trials,
                                data_fracs=data_fracs,
                                n_workers=n_workers,
                                batch_epoch_dict=batch_epoch_dict,
                                datagen_method='resample',
                                perf_eval_fn=[probabilistic_auc, demographic_parity],#demographic_parity],multiclass_demographic_parity
                                constraint_eval_fns=[],
                                # constraint_eval_kwargs=constraint_eval_kwargs,
                                results_dir=results_dir,
                                perf_eval_kwargs=perf_eval_kwargs,
                            )
                            plot_generator.run_seldonian_experiment(verbose=verbose, model_name='FRG',validation=validation, dataset_name='Adult')
                            for baseline_model in baselines:
                                plot_generator.run_baseline_experiment(
                                    model_name=baseline_model, verbose=verbose,validation=validation, dataset_name='Adult'
                                )
                            plot_generator.make_plots(
                                fontsize=12,
                                legend_fontsize=8,
                                performance_label=['AUC', '$\Delta_{\mathrm{DP}}$ <= 0.08'],
                                prob_performance_below=[None, 0.08],
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
    parser.add_argument('--validation', help='verbose', action="store_true")
    parser.add_argument('--device', help='device id', default=0)

    args = parser.parse_args()

    # constraint = args.constraint
    # epsilon = float(args.epsilon)
    # n_trials = int(args.n_trials)
    # n_workers = int(args.n_workers)
    include_baselines = args.include_baselines
    verbose = args.verbose
    validation = args.validation
    device_id = int(args.device)

    if include_baselines:
        baselines = ["ICVAE"]#,"ICVAE","VFAE", "VAE"LMIFR"controllable_vfae"] # icvae_baseline, vfae, controllable_vfae, lmifr
    else:
        baselines = []

    performance_metric="dp"

    results_base_dir = f"./SeldonianExperimentResults"
    dataset = ADULTS
    vfae_example(
        spec_rootdir="./SeldonianExperimentSpecs/vfae/spec", # icvae
        results_base_dir=results_base_dir,
        # constraints = [constraint],
        # epsilons=[epsilon],
        # n_trials=n_trials,
        performance_metric=performance_metric,
        dataset = dataset,
        baselines = baselines,
        validation = validation,
        device_id=device_id
    )
    
