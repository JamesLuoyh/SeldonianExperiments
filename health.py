# tensorflow_mnist.py
import autograd.numpy as np   # Thinly-wrapped version of Numpy
import pandas as pd
import os

from seldonian.spec import SupervisedSpec
from seldonian.dataset import SupervisedDataSet
from seldonian.utils.io_utils import load_pickle,save_pickle
from seldonian.models import objectives
from seldonian.models.pytorch_vae import PytorchVFAE
from seldonian.seldonian_algorithm import SeldonianAlgorithm
from seldonian.parse_tree.parse_tree import (
	make_parse_trees_from_constraints)
import torch

sub_regime = "classification"
print("Loading features,labels,sensitive_attrs from file...")


savename_features = '/media/yuhongluo/health/features.pkl'
savename_gender_labels = '/media/yuhongluo/health/gender_labels.pkl'
savename_mortal_labels = '/media/yuhongluo/health/mortal_labels.pkl'
savename_sensitive_attrs = '/media/yuhongluo/health/sensitive_attrs.pkl'
save_dir = "/media/yuhongluo/SeldonianExperimentSpecs/vfae/spec/"

features = load_pickle(savename_features)
# age_labels = load_pickle(savename_labels)
gender_labels = load_pickle(savename_gender_labels)
mortal_labels = load_pickle(savename_mortal_labels)
sensitive_attrs = load_pickle(savename_sensitive_attrs)
print(features.shape)
print(sensitive_attrs.shape)
# print(labels.shape)

frac_data_in_safety = 0.5
sensitive_col_names = ['0','1', '2', '3', '4', '5', '6', '7', '8', '9']

meta_information = {}
meta_information['feature_col_names'] = ['img']
meta_information['label_col_names'] = ['label']
meta_information['sensitive_col_names'] = sensitive_col_names
meta_information['sub_regime'] = sub_regime
meta_information['self_supervised'] = True
print("Making SupervisedDataSet...")
dataset = SupervisedDataSet(
    features=np.concatenate((features,sensitive_attrs,np.expand_dims(gender_labels, axis=1)), -1),
    labels=gender_labels,
    sensitive_attrs=sensitive_attrs,
    num_datapoints=features.shape[0],
    meta_information=meta_information)
regime='supervised_learning'
epsilon = 10
constraint_strs = [f'VAE <= {epsilon}']
deltas = [0.01] 
print("Making parse trees for constraint(s):")
print(constraint_strs," with deltas: ", deltas)
parse_trees = make_parse_trees_from_constraints(
    constraint_strs,deltas,regime=regime,
    sub_regime=sub_regime,columns=sensitive_col_names)
z_dim = 80
device = torch.device(1)
model = PytorchVFAE(device, **{"x_dim": features.shape[1],
        "s_dim": sensitive_attrs.shape[1],
        "y_dim": 1,
        "z1_enc_dim": z_dim,
        "z2_enc_dim": z_dim,
        "z1_dec_dim": z_dim,
        "x_dec_dim": z_dim,
        "z_dim": z_dim,
        "dropout_rate": 0.0,
        "alpha_adv": 1e-3})
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
        'alpha_theta'   : 1e-4,
        'alpha_lamb'    : 1e-4,
        'beta_velocity' : 0.9,
        'beta_rmsprop'  : 0.95,
        'use_batches'   : True,
        'batch_size'    : 200, #237
        'n_epochs'      : 80,
        'gradient_library': "autograd",
        'hyper_search'  : None,
        'verbose'       : True,
        'downstream_lr' : 1e-4,
        'downstream_bs'     : 200,
        'downstream_epochs' : 20,
        'y_dim'             : 1,
        'z_dim'             : z_dim,
        'epsilon'           : epsilon,
        'n_adv_rounds'      : 1,
        's_dim'             : sensitive_attrs.shape[1]
    },
    
    batch_size_safety=200
)
spec_save_name = os.path.join(
  save_dir, f"vfae_unsupervised_health_mutual_information_{epsilon}.pkl"
)
save_pickle(spec_save_name, spec)
print(f"Saved Spec object to: {spec_save_name}")
# save_pickle('vfae_facial_recog_{epsilon}_spec.pkl',spec,verbose=True)
# SA = SeldonianAlgorithm(spec)
# passed_safety,solution = SA.run(debug=True,write_cs_logfile=True)
# if passed_safety:
#     print("Passed safety test.")
# else:
#     print("Failed safety test")
# st_primary_objective = SA.evaluate_primary_objective(theta=solution,
#     branch='safety_test')
# print("Primary objective evaluated on safety test:")
# print(st_primary_objective)

# parse_trees[0].evaluate_constraint(theta=model.get_model_params,dataset=dataset,
# model=model.to("cpu"),regime='supervised_learning',
# branch='safety_test')
# print("VAE constraint", parse_trees[0].root.value)