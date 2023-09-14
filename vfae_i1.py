# pytorch_mnist.py 
import autograd.numpy as np   # Thinly-wrapped version of Numpy
import os

from seldonian.spec import SupervisedSpec
from seldonian.dataset import SupervisedDataSet
from seldonian.dataset import DataSetLoader
from seldonian.models.pytorch_vae import PytorchVFAE
from seldonian.models import objectives
from seldonian.seldonian_algorithm import SeldonianAlgorithm
from seldonian.parse_tree.parse_tree import (
    make_parse_trees_from_constraints)
from seldonian.utils.io_utils import load_json, save_pickle, load_pickle

import torch

ADULTS = "adults"
GERMAN = "german"


save_dir = "/media/yuhongluo/SeldonianExperimentSpecs/vfae/spec"
os.makedirs(save_dir, exist_ok=True)

if __name__ == "__main__":
    torch.manual_seed(2023)
    dataname = ADULTS
    if dataname == ADULTS:
        data_pth = "../SeldonianEngine/static/datasets/supervised/adults_vfae/vfae_adults.csv"
        metadata_pth = "../SeldonianEngine/static/datasets/supervised/adults_vfae/metadata_vfae.json"
        x_dim = 117
        z_dim = 50
        bs = 150
        s_num=2
        nce_size=50
        n_epochs = 150
        lr = 1e-4
    elif dataname == GERMAN:
        data_pth = "../SeldonianEngine/static/datasets/supervised/german_credit/vfae_german.csv"
        metadata_pth = "../SeldonianEngine/static/datasets/supervised/german_credit/metadata_vfae.json"
        x_dim = 57
        z_dim = 25
        bs = 150
        s_num=2
        nce_size=50
        n_epochs = 150
        lr = 1e-4
    else:
        raise NotImplementedError
    
    save_base_dir = 'interface_outputs'
    # save_base_dir='.'
    # Load metadata
    regime='supervised_learning'
    sub_regime='classification'

    loader = DataSetLoader(
        regime=regime)

    dataset = loader.load_supervised_dataset(
        filename=data_pth,
        metadata_filename=metadata_pth,
        file_type='csv')

    # constraint_strs = ['abs((FNR | [M]) - (FNR | [F])) <= 0.02']
    epsilon = 100
    constraint_strs = [f'VAE <= {epsilon}']
    deltas = [0.1] 
    columns = ["M", "F"]
    parse_trees = make_parse_trees_from_constraints(
        constraint_strs,deltas,regime=regime,
        sub_regime=sub_regime, columns=columns)
    device = torch.device(0)
    model = PytorchVFAE(device, **{"x_dim": x_dim,
        "s_dim": 1,
        "y_dim": 1,
        "z1_enc_dim": 100,
        "z2_enc_dim": 100,
        "z1_dec_dim": 100,
        "x_dec_dim": 100,
        "z_dim": z_dim,
        "dropout_rate": 0.0,
        "alpha_adv": lr,
        "mi_version": 1}
    )

    initial_solution_fn = model.get_model_params
    frac_data_in_safety = 0.5
    dataset.meta_information["self_supervised"] = True
    spec = SupervisedSpec(
        dataset=dataset,
        model=model,
        parse_trees=parse_trees,
        frac_data_in_safety=frac_data_in_safety,
        primary_objective=objectives.vae_loss,
        use_builtin_primary_gradient_fn=False,
        sub_regime=sub_regime,
        initial_solution_fn=initial_solution_fn,
        optimization_technique='gradient_descent',
        optimizer='adam',
        optimization_hyperparams={
            'lambda_init'   : np.array([0.0]),
            'alpha_theta'   : 1e-4,
            'alpha_lamb'    : 1e-3,
            'beta_velocity' : 0.9,
            'beta_rmsprop'  : 0.95,
            'use_batches'   : True,
            'batch_size'    : bs,
            'n_epochs'      : n_epochs,
            # 'num_iters'     : 30,
            'gradient_library': "autograd",
            'hyper_search'  : None,
            'verbose'       : True,
            'x_dim'         : x_dim,
            'z_dim'         : z_dim,
            'lr'            : lr,
            'epsilon'       : epsilon,
            'downstream_lr' : 1e-4,
            'downstream_bs'     : 100,
            'downstream_epochs' : 5,
            'y_dim'             : 1,
            'n_adv_rounds'      : 5,
        },
        batch_size_safety=200
    )
    spec_save_name = os.path.join(
        save_dir, f"vfae_unsupervised_{dataname}_mutual_information_{epsilon}.pkl"
    )
    save_pickle(spec_save_name, spec)
    print(f"Saved Spec object to: {spec_save_name}")
    
    # SA = SeldonianAlgorithm(spec)
    # passed_safety,solution = SA.run(debug=False,write_cs_logfile=True)
    # if passed_safety:
    #     print("Passed safety test.")
    # else:
    #     print("Failed safety test")
    # st_primary_objective = SA.evaluate_primary_objective(theta=solution,
    #     branch='safety_test')
    # print("Primary objective evaluated on safety test:")
    # print(st_primary_objective)

    # parse_trees[0].evaluate_constraint(theta=model.get_model_params,dataset=dataset,
    # model=model,regime='supervised_learning',
    # branch='safety_test')
    # print("VAE constraint", parse_trees[0].root.value)