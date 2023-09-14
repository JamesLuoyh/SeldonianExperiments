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

savename_features = '/media/yuhongluo/face_recog/features.pkl'
savename_gender_labels = '/media/yuhongluo/face_recog/gender_labels.pkl'
savename_age_labels = '/media/yuhongluo/face_recog/age_labels.pkl'
savename_sensitive_attrs = '/media/yuhongluo/face_recog/sensitive_attrs.pkl'
save_dir = "/media/yuhongluo/SeldonianExperimentSpecs/vfae/spec/"
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
epsilon = 0.05
constraint_strs = [f'VAE <= {epsilon}']
deltas = [0.1]
print("Making parse trees for constraint(s):")
print(constraint_strs," with deltas: ", deltas)
parse_trees = make_parse_trees_from_constraints(
    constraint_strs,deltas,regime=regime,
    sub_regime=sub_regime,columns=sensitive_col_names)
z_dim = 80
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
        'alpha_theta'   : 1e-3,
        'alpha_lamb'    : 1e-2,
        'beta_velocity' : 0.1,
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
    
    batch_size_safety=237*2
)
spec_save_name = os.path.join(
  save_dir, f"unsupervised_age_cnn_vfae_mutual_information_{epsilon}.pkl"
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