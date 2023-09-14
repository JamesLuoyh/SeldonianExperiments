# run_engine.py
from seldonian.spec import SupervisedSpec
from seldonian.dataset import SupervisedDataSet
from seldonian.utils.io_utils import load_pickle,save_pickle
from experiments.baselines.facial_recog_cnn import PytorchFacialRecog
from seldonian.models import objectives
from seldonian.seldonian_algorithm import SeldonianAlgorithm
from seldonian.parse_tree.parse_tree import (
    make_parse_trees_from_constraints)

sub_regime = "classification"
N=23700
print("Loading features,labels,sensitive_attrs from file...")
features = load_pickle(savename_features)
labels = load_pickle(savename_labels)
sensitive_attrs = load_pickle(savename_sensitive_attrs)

assert len(features) == N
assert len(labels) == N
assert len(sensitive_attrs) == N
frac_data_in_safety = 0.5
sensitive_col_names = ['M','F']

meta_information = {}
meta_information['feature_col_names'] = ['img']
meta_information['label_col_names'] = ['label']
meta_information['sensitive_col_names'] = sensitive_col_names
meta_information['sub_regime'] = sub_regime
print("Making SupervisedDataSet...")
dataset = SupervisedDataSet(
    features=features,
    labels=labels,
    sensitive_attrs=sensitive_attrs,
    num_datapoints=N,
    meta_information=meta_information)

constraint_strs = ['min((ACC | [M])/(ACC | [F]),(ACC | [F])/(ACC | [M])) >= 0.8']
deltas = [0.05] 
print("Making parse trees for constraint(s):")
print(constraint_strs," with deltas: ", deltas)
parse_trees = make_parse_trees_from_constraints(
    constraint_strs,deltas,regime=regime,
    sub_regime=sub_regime,columns=sensitive_col_names)

# Put on Mac M1 GPU via Metal performance shader (MPS) device.
# For NVIDIA graphics cards use "cuda" as the device string.
device = torch.device("mps")
model = PytorchFacialRecog(device)

initial_solution_fn = model.get_model_params
spec = SupervisedSpec(
    dataset=dataset,
    model=model,
    parse_trees=parse_trees,
    frac_data_in_safety=frac_data_in_safety,
    primary_objective=objectives.binary_logistic_loss,
    use_builtin_primary_gradient_fn=False,
    sub_regime=sub_regime,
    initial_solution_fn=initial_solution_fn,
    optimization_technique='gradient_descent',
    optimizer='adam',
    optimization_hyperparams={
        'lambda_init'   : np.array([0.5]),
        'alpha_theta'   : 0.001,
        'alpha_lamb'    : 0.001,
        'beta_velocity' : 0.9,
        'beta_rmsprop'  : 0.95,
        'use_batches'   : True,
        'batch_size'    : 237,
        'n_epochs'      : 40,
        'gradient_library': "autograd",
        'hyper_search'  : None,
        'verbose'       : True,
    },
    
    batch_size_safety=2000
)
save_pickle('/media/yuhongluo/SeldonianExperimentSpecs/vfae/spec.pkl',spec,verbose=True)
SA = SeldonianAlgorithm(spec)
passed_safety,solution = SA.run(debug=True,write_cs_logfile=True)