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
from experiments.utils import probabilistic_accuracy, probabilistic_auc, multiclass_demographic_parity, f1_score
from seldonian.utils.io_utils import load_pickle
from sklearn.model_selection import train_test_split
from seldonian.dataset import SupervisedDataSet
from seldonian.models.pytorch_cnn_vfae import PytorchFacialVAE

import torch

def vfae_example(
    spec_rootdir,
    results_base_dir,
    constraints = [],
    epsilons=[1.18],# 0.45 0.01, , 0.02],
    n_trials=1,
    data_fracs=np.logspace(-3,0,5),
    baselines = [],
    performance_metric="auc",
    n_workers=1,
    version="0",
    validation=False,
    device_id=0
):  
    data_fracs = [0.1,0.15,0.25,0.40,0.65,1]##1.0,0.65, 0.40, 0.25, 0.15,0.1]#0.40]#,0.40]#, 1, 0.40]#,0.65, 0.40, 0.25, 0.15,0.1]#,  0.1][1, 0.65, 0.40, 0.25, 0.15, 0.1]#[0.5] #  0.1,0.15,0.25,0.40,0.65,1 [0.01, 0.025, , 0.15, 0.5, 1]  # 0.01, 0.025, 0.06, 0.15, 0.5, #0.001,0.01,0.05,0.1,0.33,0.66,1
    # for baseline
    # epsilon 0.4. MI 1.
    
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
    if performance_metric == "f1":
        perf_eval_fn = f1_score
    elif performance_metric == "auc":
        perf_eval_fn = probabilistic_auc
    elif performance_metric == "accuracy":
        perf_eval_fn = probabilistic_accuracy
    elif performance_metric == "dp":
        perf_eval_fn = multiclass_demographic_parity
    # else:
    #     raise NotImplementedError(
    #         "Performance metric must be 'auc' or 'accuracy' or 'dp' for this example")
    if version == '0':
        version_subs = "_1" # baseline does not need a spec, use a random one.
    else:
        version_subs = "_" + version

    z_dim = 100
    device = torch.device(device_id)
    model = PytorchFacialVAE(device, **{"x_dim": 1,
            "s_dim": 5,
            "y_dim": 1,
            "z1_enc_dim": z_dim,
            "z2_enc_dim": z_dim,
            "z1_dec_dim": z_dim,
            "x_dec_dim": z_dim,
            "z_dim": z_dim,
            "dropout_rate": 0.0,
            "alpha_adv": 1e-3,
            "mi_version": 1
            })
    
    print("version", version)
    for epsilon in epsilons:
        specfile = os.path.join(
            spec_rootdir,
            f"unsupervised_cnn_vfae{version_subs}_mutual_information_{epsilon}.pkl"
        )
        print(f"unsupervised_cnn_vfae{version_subs}_mutual_information_{epsilon}.pkl")
        spec = load_pickle(specfile)
        spec.model = model
        torch.manual_seed(2023) #spec.seed)
        np.random.seed(2023) #spec.seed)
        # if epsilon == 0.4:
        #     batch_epoch_dict = {
        #     0.1:[237*2,250],
        #     0.15:[237*2,167],
        #     0.25:[237*2, 100],
        #     0.40:[237*2, 63], #150 (1e-4), #50 (1e-3)
        #     0.65:[237*2, 38], #150 (1e-4), #50 (1e-3)
        #     0.5:[237*2,50], #150
        #     1.0: [237*2,25], #100 (1e-4), #50 (1e-3)
        #     }
        # elif epsilon == 0.1:
        #     # epsilon 0.1. MI 1.
        #     batch_epoch_dict = {
        #     0.1:[237*2,250],
        #     0.15:[237*2,167],
        #     0.25:[237*2, 100],
        #     0.40:[237*2, 63], #150 (1e-4), #50 (1e-3)
        #     0.65:[237*2, 38], #150 (1e-4), #50 (1e-3)
        #     0.5:[237*2,50], #150
        #     1.0: [237*2,60], #100 (1e-4), #50 (1e-3)
        #     }
        # elif epsilon == 0.05:
        #     # epsilon 0.05. MI 2.
        #     batch_epoch_dict = {
        #     0.1:[237*2,250],
        #     0.15:[237*2,167],
        #     0.25:[237*2, 100],
        #     0.40:[237*2, 63], #150 (1e-4), #50 (1e-3)
        #     0.65:[237*2, 38], #150 (1e-4), #50 (1e-3)
        #     0.5:[237*2,50], #150
        #     1.0: [237*2,60], #100 (1e-4), #50 (1e-3)
        #     }
        #1.18
        alpha_l = [1e-4]
        alpha_lambda_l = [1e-3]
        lambda_init_l = [.1]
        epochs_l = [30]
        delta_l = [0.1]
        # alpha_l = [1e-4]#, 1e-5] #[1e-4, 1e-5]# [1e-4, 1e-5] #[1e-3, 1e-4]
        # alpha_lambda_l = [1e-3]#[1e-3]#[1e-3,1e-4]#[1e-3, 1e-4] #[1e-2] #[1e-3, 1e-4]
        # lambda_init_l = [.1]#, 1.0, 0.5]#[1.0]#[0.05,0.1,0.2]#[.01, 0.1, 0.5, 1.0]#[1e-2]#[1e-1, 1e-2]#[0.1, 0.15,0.25,0.40,0.65, 1]
        # epochs_l = [30]#, 60, 90]#30]#[60,90]#   , 90]#[150]#, 120, 150]#[175]# [200, 250]#, 100, 125, 150]#50, 75] #, 100#, 125, 150]
        # delta_l = [0.7]#[0.5,0.7,0.9]#, 0.7, 0.9]
        #delta_dp,mi,auc,epsilon,lagrange,lr,epochs,lrl
        #0.078751758740609,2.1635162830352783,0.6380527917052579,10.0,1.0,0.001,10.0,0.0001
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
                                0.1:[237*2,int(epochs/0.1)],
                                0.15: [237*2,int(epochs/0.15)],
                                0.25:[237*2,int(epochs/0.25)],
                                0.40:[237*2,int(epochs/0.4)],
                                0.65:[237*2,int(epochs/0.65)],
                                1.0: [237*2,epochs],
                            }

                            if validation:
                                suffix = "validation"
                            else:
                                suffix = "test"
                            results_dir = os.path.join(results_base_dir,
                                f"cnn_vfae_mutual_information_{epsilon}_{alpha}_{alpha_lambda}_{lambda_init}_{epochs}_{delta}_{suffix}")

                            orig_features = spec.dataset.features
                            orig_features_X, orig_features_S, orig_features_Y = orig_features
                            orig_labels_gender = spec.dataset.labels[0]
                            orig_labels_age = spec.dataset.labels[1]
                            orig_sensitive_attrs = spec.dataset.sensitive_attrs
                            # First, shuffle features
                            (train_features_X,test_features_X,train_features_S, test_features_S, 
                            train_features_Y, test_features_Y, train_gender_labels,
                            test_gender_labels,train_age_labels, test_age_labels,train_sensitive_attrs,
                            test_sensitive_attrs
                                ) = train_test_split(
                                    orig_features_X,
                                    orig_features_S,
                                    orig_features_Y,
                                    orig_labels_gender,
                                    orig_labels_age,
                                    orig_sensitive_attrs,
                                    shuffle=True,
                                    test_size=0.2,
                                    random_state=42)
                            new_dataset = SupervisedDataSet(
                            features=[train_features_X, train_features_S, train_features_Y], 
                            labels=[train_gender_labels, train_age_labels],
                            sensitive_attrs=train_sensitive_attrs, 
                            num_datapoints=len(train_features_X),
                            meta_information=spec.dataset.meta_information)
                            # Set spec dataset to this new dataset
                            spec.dataset = new_dataset
                            # Setup performance evaluation function and kwargs 
                            perf_eval_kwargs = {
                                'X':[test_features_X, test_features_S, test_features_Y],
                                'y':[test_gender_labels, test_age_labels],
                                'performance_metric':['auc', 'dp'],
                                'device': torch.device(device_id),
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
                                perf_eval_fn=[probabilistic_auc, multiclass_demographic_parity],
                                constraint_eval_fns=[],
                                results_dir=results_dir,
                                perf_eval_kwargs=perf_eval_kwargs,
                                n_downstreams=2,
                            )
                            if version != '0':
                                plot_generator.run_seldonian_experiment(verbose=verbose,model_name='FRG',validation=validation, dataset_name='Face')
                            else:
                                for baseline_model in baselines:
                                    plot_generator.run_baseline_experiment(
                                        model_name=baseline_model, verbose=verbose,validation=validation, dataset_name='Face'
                                    )
                            dp_constraint = 0.08
                            for i in range(2):
                                plot_savename = os.path.join(
                                    results_dir, f"cnn_vfae{version_subs}_mutual_information_{epsilon}_{performance_metric}_downstream_{i}.pdf"
                                )
                                plot_generator.make_plots(
                                    fontsize=12,
                                    legend_fontsize=8,
                                    performance_label=['auc', '$\Delta_{\mathrm{DP}}$ <= 0.08'],
                                    prob_performance_below=[None, dp_constraint],
                                    performance_yscale="linear",
                                    savename=plot_savename,
                                    result_filename_suffix=f"_downstream_{i}"
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
    parser.add_argument('--version', help='which frg version to use. 0 uses baselines, 1 uses I1, 2 uses I2, 3 uses I1 and I2', required=True)
    parser.add_argument('--validation', help='validation', action="store_true")
    parser.add_argument('--device', help='device id', default=0)

    args = parser.parse_args()

    # constraint = args.constraint
    # epsilon = float(args.epsilon)
    # n_trials = int(args.n_trials)
    # n_workers = int(args.n_workers)
    include_baselines = args.include_baselines
    verbose = args.verbose
    version = args.version
    validation = args.validation
    device_id = int(args.device)

    # if include_baselines:
    baselines = ["cnn_lmifr_all"] #"cnn_controllable_vfae","cnn_icvae" "cnn_lmifr_all", "cnn_vfae_baseline", "cnn_vae"
    # else:
    #     baselines = []

    performance_metric="auc_dp"

    results_base_dir = f"./SeldonianExperimentResults"
    vfae_example(
        spec_rootdir="../SeldonianExperimentSpecs/vfae/spec",
        results_base_dir=results_base_dir,
        # constraints = [constraint],
        # epsilons=[epsilon],
        # n_trials=n_trials,
        performance_metric=performance_metric,
        baselines = baselines,
        version = version,
        validation=validation,
        device_id=device_id
    )
    
