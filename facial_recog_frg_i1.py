# tensorflow_mnist.py
import autograd.numpy as np   # Thinly-wrapped version of Numpy
import pandas as pd
import os

from seldonian.spec import SupervisedSpec
from seldonian.dataset import SupervisedDataSet
from seldonian.utils.io_utils import load_pickle,save_pickle
from seldonian.models import objectives
from seldonian.models.pytorch_cnn_vfae import PytorchFacialVAE
from seldonian.seldonian_algorithm import SeldonianAlgorithm
from seldonian.parse_tree.parse_tree import (
	make_parse_trees_from_constraints)
import torch

sub_regime = "classification"
N=23700
print("Loading features,labels,sensitive_attrs from file...")

savename_features = '../face_recog/features.pkl'
savename_gender_labels = '../face_recog/gender_labels.pkl'
savename_age_labels = '../face_recog/age_labels.pkl'
savename_sensitive_attrs = '../face_recog/sensitive_attrs.pkl'
save_dir = "../SeldonianExperimentSpecs/vfae/spec/"
features = load_pickle(savename_features)
age_labels = load_pickle(savename_age_labels)
gender_labels = load_pickle(savename_gender_labels)
sensitive_attrs = load_pickle(savename_sensitive_attrs)
print(features.shape)
print(sensitive_attrs.shape)
# print(labels.shape)
# assert len(features) == N
# assert len(gender_labels) == N
# assert len(age_labels) == N
# assert len(sensitive_attrs) == N
frac_data_in_safety = 0.5
sensitive_col_names = ['0','1', '2', '3', '4']

meta_information = {}
meta_information['feature_col_names'] = ['img']
meta_information['label_col_names'] = ['label']
meta_information['sensitive_col_names'] = sensitive_col_names
meta_information['sub_regime'] = sub_regime
meta_information['self_supervised'] = True
print("Making SupervisedDataSet...")
dataset = SupervisedDataSet(
    features=[features,sensitive_attrs, gender_labels],
    labels=[gender_labels, age_labels],
    sensitive_attrs=sensitive_attrs,
    num_datapoints=N,
    meta_information=meta_information)
regime='supervised_learning'
# epsilon = 0.4
# deltas = [0.1]
epsilon = 0.15#1.18
deltas = [0.7]

constraint_strs = [f'VAE <= {epsilon}']

print("Making parse trees for constraint(s):")
print(constraint_strs," with deltas: ", deltas)
parse_trees = make_parse_trees_from_constraints(
    constraint_strs,deltas,regime=regime,
    sub_regime=sub_regime,columns=sensitive_col_names)
z_dim = 100
device = torch.device(0)
model = PytorchFacialVAE(device, **{"x_dim": features.shape[1],
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
        
lambda_init = 1.0
initial_solution_fn = model.get_model_params
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
        'lambda_init'   : np.array([lambda_init]),
        'alpha_theta'   : 1e-3,
        'alpha_lamb'    : 1e-2,
        'beta_velocity' : 0.6,
        'beta_rmsprop'  : 0.95,
        'use_batches'   : True,
        'batch_size'    : 237, #237
        'n_epochs'      : 80,
        'gradient_library': "autograd",
        'hyper_search'  : None,
        'verbose'       : True,
        'downstream_lr' : 1e-4,
        'downstream_bs'     : 237,
        'downstream_epochs' : 10,
        'y_dim'             : 1,
        'z_dim'             : z_dim,
        'epsilon'           : epsilon,
        'n_adv_rounds'      : 3,
        's_dim'             : sensitive_attrs.shape[1]
    },
    
    batch_size_safety=237
)
spec_save_name = os.path.join(
  save_dir, f"unsupervised_cnn_vfae_1_mutual_information_{epsilon}.pkl"
)
save_pickle(spec_save_name, spec)
print(f"Saved Spec object to: {spec_save_name}")
