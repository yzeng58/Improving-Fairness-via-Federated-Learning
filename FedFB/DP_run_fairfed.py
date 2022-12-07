import numpy as np
from utils import *
import torch


# Adult
sensitive_attributes = ['race']
categorical_attributes = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'sex', 'native-country']
continuous_attributes = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
features_to_keep = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
            'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss','hours-per-week', 
            'native-country', 'salary']
label_name = 'salary'

adult = process_csv('adult', 'adult.data', label_name, ' >50K', sensitive_attributes, [' White'], categorical_attributes, continuous_attributes, features_to_keep, na_values = [], header = None, columns = features_to_keep)
test = process_csv('adult', 'adult.test', label_name, ' >50K.', sensitive_attributes, [' White'], categorical_attributes, continuous_attributes, features_to_keep, na_values = [], header = None, columns = features_to_keep) # the distribution is very different from training distribution
test['native-country_ Holand-Netherlands'] = 0
test = test[adult.columns]

np.random.seed(1)
random.seed(1)
adult_white_idx = adult[adult['z'] == 1].index
adult_others_idx = adult[adult['z'] == 0].index
adult_mean_sensitive = adult['z'].mean()

client1_idx = np.concatenate((np.random.choice(adult_others_idx, 269, replace = True),
                                np.random.choice(adult_white_idx, 615, replace = True)))
client2_idx = np.concatenate((np.random.choice(adult_others_idx, 128, replace = True),
                                np.random.choice(adult_white_idx, 29839, replace = True)))
client3_idx = np.concatenate((np.random.choice(adult_others_idx, 418, replace = True),
                                np.random.choice(adult_white_idx, 74, replace = True)))
client4_idx = np.concatenate((np.random.choice(adult_others_idx, 43, replace = True),
                                np.random.choice(adult_white_idx, 392, replace = True)))
client5_idx = np.concatenate((np.random.choice(adult_others_idx, 4196, replace = True),
                                np.random.choice(adult_white_idx, 203, replace = True)))
random.shuffle(client1_idx)
random.shuffle(client2_idx)
random.shuffle(client3_idx)
random.shuffle(client4_idx)
random.shuffle(client5_idx)

adult_clients_idx = [client1_idx, client2_idx, client3_idx, client4_idx, client5_idx]

adult_num_features = len(adult.columns)-1
adult_test = LoadData(test, 'salary', 'z')
adult_train = LoadData(adult, 'salary', 'z')
torch.manual_seed(0)
adult_info = [adult_train, adult_test, adult_clients_idx]

# Adult-alpha 0.2

adult_white_len = len(adult_white_idx)
adult_others_len = len(adult_others_idx)

alpha = 0.2

client1_portion = np.random.dirichlet((alpha,alpha))
client1_idx = np.concatenate((np.random.choice(adult_white_idx, max(int(client1_portion[0] * adult_white_len),100), replace = True),
                                np.random.choice(adult_others_idx, int(client1_portion[0] * adult_others_len), replace = True)))

client2_portion = np.random.dirichlet((alpha,alpha))
client2_idx = np.concatenate((np.random.choice(adult_white_idx, max(int(client2_portion[0] * adult_white_len),100), replace = True),
                                np.random.choice(adult_others_idx, int(client2_portion[0] * adult_others_len), replace = True)))

client3_portion = np.random.dirichlet((alpha,alpha))
client3_idx = np.concatenate((np.random.choice(adult_white_idx, max(int(client3_portion[0] * adult_white_len),100), replace = True),
                                np.random.choice(adult_others_idx, int(client3_portion[0] * adult_others_len), replace = True)))

client4_portion = np.random.dirichlet((alpha,alpha))
client4_idx = np.concatenate((np.random.choice(adult_white_idx, max(int(client4_portion[0] * adult_white_len),100), replace = True),
                                np.random.choice(adult_others_idx, int(client4_portion[0] * adult_others_len), replace = True)))

client5_portion = np.random.dirichlet((alpha,alpha))
client5_idx = np.concatenate((np.random.choice(adult_white_idx, max(int(client5_portion[0] * adult_white_len),100), replace = True),
                                np.random.choice(adult_others_idx, int(client5_portion[0] * adult_others_len), replace = True)))

random.shuffle(client1_idx)
random.shuffle(client2_idx)
random.shuffle(client3_idx)
random.shuffle(client4_idx)
random.shuffle(client5_idx)

adult_clients_idx = [client1_idx, client2_idx, client3_idx, client4_idx, client5_idx]
adult_alpha_0_2_info = [adult_train, adult_test, adult_clients_idx]

# Adult-alpha 0.5

adult_white_len = len(adult_white_idx)
adult_others_len = len(adult_others_idx)

alpha = 0.5

client1_portion = np.random.dirichlet((alpha,alpha))
client1_idx = np.concatenate((np.random.choice(adult_white_idx, max(int(client1_portion[0] * adult_white_len),100), replace = True),
                                np.random.choice(adult_others_idx, int(client1_portion[0] * adult_others_len), replace = True)))

client2_portion = np.random.dirichlet((alpha,alpha))
client2_idx = np.concatenate((np.random.choice(adult_white_idx, max(int(client2_portion[0] * adult_white_len),100), replace = True),
                                np.random.choice(adult_others_idx, int(client2_portion[0] * adult_others_len), replace = True)))

client3_portion = np.random.dirichlet((alpha,alpha))
client3_idx = np.concatenate((np.random.choice(adult_white_idx, max(int(client3_portion[0] * adult_white_len),100), replace = True),
                                np.random.choice(adult_others_idx, int(client3_portion[0] * adult_others_len), replace = True)))

client4_portion = np.random.dirichlet((alpha,alpha))
client4_idx = np.concatenate((np.random.choice(adult_white_idx, max(int(client4_portion[0] * adult_white_len),100), replace = True),
                                np.random.choice(adult_others_idx, int(client4_portion[0] * adult_others_len), replace = True)))

client5_portion = np.random.dirichlet((alpha,alpha))
client5_idx = np.concatenate((np.random.choice(adult_white_idx, max(int(client5_portion[0] * adult_white_len),100), replace = True),
                                np.random.choice(adult_others_idx, int(client5_portion[0] * adult_others_len), replace = True)))

random.shuffle(client1_idx)
random.shuffle(client2_idx)
random.shuffle(client3_idx)
random.shuffle(client4_idx)
random.shuffle(client5_idx)

adult_clients_idx = [client1_idx, client2_idx, client3_idx, client4_idx, client5_idx]
adult_alpha_0_5_info = [adult_train, adult_test, adult_clients_idx]

# Adult-alpha 10
client1_idx = np.concatenate((np.random.choice(adult_white_idx, 3585, replace = True),
                                np.random.choice(adult_others_idx, 1505, replace = True)))
client2_idx = np.concatenate((np.random.choice(adult_white_idx, 5695, replace = True),
                                np.random.choice(adult_others_idx, 876, replace = True)))
client3_idx = np.concatenate((np.random.choice(adult_white_idx, 7261, replace = True),
                                np.random.choice(adult_others_idx, 978, replace = True)))
client4_idx = np.concatenate((np.random.choice(adult_white_idx, 5848, replace = True),
                                np.random.choice(adult_others_idx, 601, replace = True)))
client5_idx = np.concatenate((np.random.choice(adult_white_idx, 8734, replace = True),
                                np.random.choice(adult_others_idx, 1094, replace = True)))

random.shuffle(client1_idx)
random.shuffle(client2_idx)
random.shuffle(client3_idx)
random.shuffle(client4_idx)
random.shuffle(client5_idx)

adult_clients_idx = [client1_idx, client2_idx, client3_idx, client4_idx, client5_idx]
adult_alpha_10_info = [adult_train, adult_test, adult_clients_idx]

# Adult-alpha 5000

adult_white_len = len(adult_white_idx)
adult_others_len = len(adult_others_idx)

alpha = 5000

client1_portion = np.random.dirichlet((alpha,alpha))
client1_idx = np.concatenate((np.random.choice(adult_white_idx, max(int(client1_portion[0] * adult_white_len),100), replace = True),
                                np.random.choice(adult_others_idx, int(client1_portion[0] * adult_others_len), replace = True)))

client2_portion = np.random.dirichlet((alpha,alpha))
client2_idx = np.concatenate((np.random.choice(adult_white_idx, max(int(client2_portion[0] * adult_white_len),100), replace = True),
                                np.random.choice(adult_others_idx, int(client2_portion[0] * adult_others_len), replace = True)))

client3_portion = np.random.dirichlet((alpha,alpha))
client3_idx = np.concatenate((np.random.choice(adult_white_idx, max(int(client3_portion[0] * adult_white_len),100), replace = True),
                                np.random.choice(adult_others_idx, int(client3_portion[0] * adult_others_len), replace = True)))

client4_portion = np.random.dirichlet((alpha,alpha))
client4_idx = np.concatenate((np.random.choice(adult_white_idx, max(int(client4_portion[0] * adult_white_len),100), replace = True),
                                np.random.choice(adult_others_idx, int(client4_portion[0] * adult_others_len), replace = True)))

client5_portion = np.random.dirichlet((alpha,alpha))
client5_idx = np.concatenate((np.random.choice(adult_white_idx, max(int(client5_portion[0] * adult_white_len),100), replace = True),
                                np.random.choice(adult_others_idx, int(client5_portion[0] * adult_others_len), replace = True)))

random.shuffle(client1_idx)
random.shuffle(client2_idx)
random.shuffle(client3_idx)
random.shuffle(client4_idx)
random.shuffle(client5_idx)

adult_clients_idx = [client1_idx, client2_idx, client3_idx, client4_idx, client5_idx]
adult_alpha_5000_info = [adult_train, adult_test, adult_clients_idx]

# COMPAS
sensitive_attributes = ['sex']
categorical_attributes = ['age_cat', 'c_charge_degree', 'c_charge_desc', 'race']
continuous_attributes = ['age', 'juv_fel_count', 'juv_misd_count', 'juv_other_count', 'priors_count']
features_to_keep = ['sex', 'age', 'age_cat', 'race', 'juv_fel_count', 'juv_misd_count', 'juv_other_count',
        'priors_count', 'c_charge_degree', 'c_charge_desc','two_year_recid']
label_name = 'two_year_recid'

compas = process_csv('compas', 'compas-scores-two-years.csv', label_name, 0, sensitive_attributes, ['Male'], categorical_attributes, continuous_attributes, features_to_keep)
train = compas.iloc[:int(len(compas)*.7)]
test = compas.iloc[int(len(compas)*.7):]

np.random.seed(1)
random.seed(1)
compas_protected_idx = train[train['z'] == 1].index
compas_others_idx = train[train['z'] == 0].index
compas_mean_sensitive = train['z'].mean()

compas_protected_len = len(compas_protected_idx)
compas_others_len = len(compas_others_idx)
alpha = 0.1
client1_portion = np.random.dirichlet((alpha,alpha))
client1_idx = np.concatenate((np.random.choice(compas_protected_idx, int(client1_portion[0] * compas_protected_len), replace = True),
                                np.random.choice(compas_others_idx, int(client1_portion[1] * compas_others_len), replace = True)))
client2_portion = np.random.dirichlet((alpha,alpha))
client2_idx = np.concatenate((np.random.choice(compas_protected_idx, int(client2_portion[0] * compas_protected_len), replace = True),
                                np.random.choice(compas_others_idx, int(client2_portion[1] * compas_others_len), replace = True)))
client3_portion = np.random.dirichlet((alpha,alpha))
client3_idx = np.concatenate((np.random.choice(compas_protected_idx, int(client3_portion[0] * compas_protected_len), replace = True),
                                np.random.choice(compas_others_idx, int(client3_portion[1] * compas_others_len), replace = True)))
client4_portion = np.random.dirichlet((alpha,alpha))
client4_idx = np.concatenate((np.random.choice(compas_protected_idx, int(client4_portion[0] * compas_protected_len), replace = True),
                                np.random.choice(compas_others_idx, int(client4_portion[1] * compas_others_len), replace = True)))
client5_portion = np.random.dirichlet((alpha,alpha))
client5_idx = np.concatenate((np.random.choice(compas_protected_idx, int(client5_portion[0] * compas_protected_len), replace = True),
                                np.random.choice(compas_others_idx, int(client5_portion[1] * compas_others_len), replace = True)))
random.shuffle(client1_idx)
random.shuffle(client2_idx)
random.shuffle(client3_idx)
random.shuffle(client4_idx)
random.shuffle(client5_idx)

compas_clients_idx = [client1_idx, client2_idx, client3_idx, client4_idx, client5_idx]
compas_z = len(set(compas.z))
compas_num_features = len(compas.columns) - 1
compas_train = LoadData(train, label_name, 'z')
compas_test = LoadData(test, label_name, 'z')

compas_info = [compas_train, compas_test, compas_clients_idx]

# COMPAS 0.2
alpha = 0.2
client1_portion = np.random.dirichlet((alpha,alpha))
client1_idx = np.concatenate((np.random.choice(compas_protected_idx, int(client1_portion[0] * compas_protected_len), replace = True),
                                np.random.choice(compas_others_idx, int(client1_portion[1] * compas_others_len), replace = True)))
client2_portion = np.random.dirichlet((alpha,alpha))
client2_idx = np.concatenate((np.random.choice(compas_protected_idx, int(client2_portion[0] * compas_protected_len), replace = True),
                                np.random.choice(compas_others_idx, int(client2_portion[1] * compas_others_len), replace = True)))
client3_portion = np.random.dirichlet((alpha,alpha))
client3_idx = np.concatenate((np.random.choice(compas_protected_idx, int(client3_portion[0] * compas_protected_len), replace = True),
                                np.random.choice(compas_others_idx, int(client3_portion[1] * compas_others_len), replace = True)))
client4_portion = np.random.dirichlet((alpha,alpha))
client4_idx = np.concatenate((np.random.choice(compas_protected_idx, int(client4_portion[0] * compas_protected_len), replace = True),
                                np.random.choice(compas_others_idx, int(client4_portion[1] * compas_others_len), replace = True)))
client5_portion = np.random.dirichlet((alpha,alpha))
client5_idx = np.concatenate((np.random.choice(compas_protected_idx, int(client5_portion[0] * compas_protected_len), replace = True),
                                np.random.choice(compas_others_idx, int(client5_portion[1] * compas_others_len), replace = True)))
random.shuffle(client1_idx)
random.shuffle(client2_idx)
random.shuffle(client3_idx)
random.shuffle(client4_idx)
random.shuffle(client5_idx)

compas_clients_idx = [client1_idx, client2_idx, client3_idx, client4_idx, client5_idx]
compas_alpha_0_2_info = [compas_train, compas_test, compas_clients_idx]

# COMPAS 0.5
alpha = 0.5
client1_portion = np.random.dirichlet((alpha,alpha))
client1_idx = np.concatenate((np.random.choice(compas_protected_idx, int(client1_portion[0] * compas_protected_len), replace = True),
                                np.random.choice(compas_others_idx, int(client1_portion[1] * compas_others_len), replace = True)))
client2_portion = np.random.dirichlet((alpha,alpha))
client2_idx = np.concatenate((np.random.choice(compas_protected_idx, int(client2_portion[0] * compas_protected_len), replace = True),
                                np.random.choice(compas_others_idx, int(client2_portion[1] * compas_others_len), replace = True)))
client3_portion = np.random.dirichlet((alpha,alpha))
client3_idx = np.concatenate((np.random.choice(compas_protected_idx, int(client3_portion[0] * compas_protected_len), replace = True),
                                np.random.choice(compas_others_idx, int(client3_portion[1] * compas_others_len), replace = True)))
client4_portion = np.random.dirichlet((alpha,alpha))
client4_idx = np.concatenate((np.random.choice(compas_protected_idx, int(client4_portion[0] * compas_protected_len), replace = True),
                                np.random.choice(compas_others_idx, int(client4_portion[1] * compas_others_len), replace = True)))
client5_portion = np.random.dirichlet((alpha,alpha))
client5_idx = np.concatenate((np.random.choice(compas_protected_idx, int(client5_portion[0] * compas_protected_len), replace = True),
                                np.random.choice(compas_others_idx, int(client5_portion[1] * compas_others_len), replace = True)))
random.shuffle(client1_idx)
random.shuffle(client2_idx)
random.shuffle(client3_idx)
random.shuffle(client4_idx)
random.shuffle(client5_idx)

compas_clients_idx = [client1_idx, client2_idx, client3_idx, client4_idx, client5_idx]
compas_alpha_0_5_info = [compas_train, compas_test, compas_clients_idx]

# COMPAS 10
alpha = 10
client1_portion = np.random.dirichlet((alpha,alpha))
client1_idx = np.concatenate((np.random.choice(compas_protected_idx, int(client1_portion[0] * compas_protected_len), replace = True),
                                np.random.choice(compas_others_idx, int(client1_portion[1] * compas_others_len), replace = True)))
client2_portion = np.random.dirichlet((alpha,alpha))
client2_idx = np.concatenate((np.random.choice(compas_protected_idx, int(client2_portion[0] * compas_protected_len), replace = True),
                                np.random.choice(compas_others_idx, int(client2_portion[1] * compas_others_len), replace = True)))
client3_portion = np.random.dirichlet((alpha,alpha))
client3_idx = np.concatenate((np.random.choice(compas_protected_idx, int(client3_portion[0] * compas_protected_len), replace = True),
                                np.random.choice(compas_others_idx, int(client3_portion[1] * compas_others_len), replace = True)))
client4_portion = np.random.dirichlet((alpha,alpha))
client4_idx = np.concatenate((np.random.choice(compas_protected_idx, int(client4_portion[0] * compas_protected_len), replace = True),
                                np.random.choice(compas_others_idx, int(client4_portion[1] * compas_others_len), replace = True)))
client5_portion = np.random.dirichlet((alpha,alpha))
client5_idx = np.concatenate((np.random.choice(compas_protected_idx, int(client5_portion[0] * compas_protected_len), replace = True),
                                np.random.choice(compas_others_idx, int(client5_portion[1] * compas_others_len), replace = True)))
random.shuffle(client1_idx)
random.shuffle(client2_idx)
random.shuffle(client3_idx)
random.shuffle(client4_idx)
random.shuffle(client5_idx)

compas_clients_idx = [client1_idx, client2_idx, client3_idx, client4_idx, client5_idx]
compas_alpha_10_info = [compas_train, compas_test, compas_clients_idx]

# COMPAS 5000
alpha = 5000
client1_portion = np.random.dirichlet((alpha,alpha))
client1_idx = np.concatenate((np.random.choice(compas_protected_idx, int(client1_portion[0] * compas_protected_len), replace = True),
                                np.random.choice(compas_others_idx, int(client1_portion[1] * compas_others_len), replace = True)))
client2_portion = np.random.dirichlet((alpha,alpha))
client2_idx = np.concatenate((np.random.choice(compas_protected_idx, int(client2_portion[0] * compas_protected_len), replace = True),
                                np.random.choice(compas_others_idx, int(client2_portion[1] * compas_others_len), replace = True)))
client3_portion = np.random.dirichlet((alpha,alpha))
client3_idx = np.concatenate((np.random.choice(compas_protected_idx, int(client3_portion[0] * compas_protected_len), replace = True),
                                np.random.choice(compas_others_idx, int(client3_portion[1] * compas_others_len), replace = True)))
client4_portion = np.random.dirichlet((alpha,alpha))
client4_idx = np.concatenate((np.random.choice(compas_protected_idx, int(client4_portion[0] * compas_protected_len), replace = True),
                                np.random.choice(compas_others_idx, int(client4_portion[1] * compas_others_len), replace = True)))
client5_portion = np.random.dirichlet((alpha,alpha))
client5_idx = np.concatenate((np.random.choice(compas_protected_idx, int(client5_portion[0] * compas_protected_len), replace = True),
                                np.random.choice(compas_others_idx, int(client5_portion[1] * compas_others_len), replace = True)))
random.shuffle(client1_idx)
random.shuffle(client2_idx)
random.shuffle(client3_idx)
random.shuffle(client4_idx)
random.shuffle(client5_idx)

compas_clients_idx = [client1_idx, client2_idx, client3_idx, client4_idx, client5_idx]
compas_alpha_5000_info = [compas_train, compas_test, compas_clients_idx]

from ray.tune.progress_reporter import CLIReporter
from DP_server import *
from ray import tune
from ray.tune.schedulers import ASHAScheduler
import pandas as pd

def run_dp(method, model, dataset, prn = True, seed = 123, trial = False, metric = "Demographic disparity", **kwargs):
    # choose the model
    if model == 'logistic regression':
        arc = logReg
    elif model == 'multilayer perceptron':
        arc = mlp
    else:
        Warning('Does not support this model!')
        exit(1)

    # set up the dataset
    if dataset == 'synthetic':
        Z, num_features, info = 2, 3, synthetic_info
    elif dataset == 'adult':
        Z, num_features, info = 2, adult_num_features, adult_info
    elif dataset == 'adult_alpha_10':
        Z, num_features, info = 2, adult_num_features, adult_alpha_10_info
    elif dataset == 'adult_alpha_0.2':
        Z, num_features, info = 2, adult_num_features, adult_alpha_0_2_info
    elif dataset == 'adult_alpha_0.5':
        Z, num_features, info = 2, adult_num_features, adult_alpha_0_5_info
    elif dataset == 'adult_alpha_5000':
        Z, num_features, info = 2, adult_num_features, adult_alpha_5000_info
    elif dataset == 'compas':
        Z, num_features, info = compas_z, compas_num_features, compas_info
    elif dataset == 'compas_alpha_0.2':
        Z, num_features, info = compas_z, compas_num_features, compas_alpha_0_2_info
    elif dataset == 'compas_alpha_0.5':
        Z, num_features, info = compas_z, compas_num_features, compas_alpha_0_5_info
    elif dataset == 'compas_alpha_10':
        Z, num_features, info = compas_z, compas_num_features, compas_alpha_10_info
    elif dataset == 'compas_alpha_5000':
        Z, num_features, info = compas_z, compas_num_features, compas_alpha_5000_info
    elif dataset == 'communities':
        Z, num_features, info = communities_z, communities_num_features, communities_info
    elif dataset == 'bank':
        Z, num_features, info = bank_z, bank_num_features, bank_info
    else:
        Warning('Does not support this dataset!')
        exit(1)

    # set up the server
    server = Server(arc(num_features=num_features, num_classes=2, seed = seed), info, train_prn = False, seed = seed, Z = Z, ret = True, prn = prn, trial = trial,  metric = metric)

    # execute
    if method == 'fedavg':
        acc, dpdisp, classifier = server.FedAvg(**kwargs)
    elif method == 'uflfb':
        acc, dpdisp, classifier = server.UFLFB(**kwargs)
    elif method == 'fedfb':
        acc, dpdisp, classifier = server.FedFB(**kwargs)
    elif method == 'cflfb':
        acc, dpdisp, classifier = server.CFLFB(**kwargs)
    elif method == 'fflfb':
        acc, dpdisp, classifier = server.FFLFB(**kwargs)
    elif method == 'fairfed':
        acc, dpdisp, classifier = server.FairFed(**kwargs)
    else:
        Warning('Does not support this method!')
        exit(1)

    if not trial: return {'accuracy': acc, 'DP Disp': dpdisp}

def sim_dp(method, model, dataset, num_sim = 5, seed = 0, metric = "Demographic disparity", resources_per_trial = {'cpu':4}, **kwargs):
    # choose the model
    if model == 'logistic regression':
        arc = logReg
    elif model == 'multilayer perceptron':
        arc = mlp
    else:
        Warning('Does not support this model!')
        exit(1)

    # set up the dataset
    if dataset == 'synthetic':
        Z, num_features, info = 2, 3, synthetic_info
    elif dataset == 'adult':
        Z, num_features, info = 2, adult_num_features, adult_info
    elif dataset == 'adult_alpha_10':
        Z, num_features, info = 2, adult_num_features, adult_alpha_10_info
    elif dataset == 'adult_alpha_0.2':
        Z, num_features, info = 2, adult_num_features, adult_alpha_0_2_info
    elif dataset == 'adult_alpha_0.5':
        Z, num_features, info = 2, adult_num_features, adult_alpha_0_5_info
    elif dataset == 'adult_alpha_5000':
        Z, num_features, info = 2, adult_num_features, adult_alpha_5000_info
    elif dataset == 'compas':
        Z, num_features, info = compas_z, compas_num_features, compas_info
    elif dataset == 'compas_alpha_0.2':
        Z, num_features, info = compas_z, compas_num_features, compas_alpha_0_2_info
    elif dataset == 'compas_alpha_0.5':
        Z, num_features, info = compas_z, compas_num_features, compas_alpha_0_5_info
    elif dataset == 'compas_alpha_10':
        Z, num_features, info = compas_z, compas_num_features, compas_alpha_10_info
    elif dataset == 'compas_alpha_5000':
        Z, num_features, info = compas_z, compas_num_features, compas_alpha_5000_info
    elif dataset == 'communities':
        Z, num_features, info = communities_z, communities_num_features, communities_info
    elif dataset == 'bank':
        Z, num_features, info = bank_z, bank_num_features, bank_info
    else:
        Warning('Does not support this dataset!')
        exit(1)

    if method == 'fedavg':
        print('--------------------------------Hyperparameter selection--------------------------------')
        print('--------------------------------Seed:' + str(seed) + '--------------------------------')
        config = {'lr': tune.grid_search([.001, .002, .005, .01, .02])}
        def trainable(config): 
            return run_dp(method = method, model = model, dataset = dataset, metric = metric, prn = False, trial = True, seed = seed, learning_rate = config['lr'], **kwargs)

        asha_scheduler = ASHAScheduler(
            time_attr = 'iteration',
            metric = 'loss',
            mode = 'min',
            grace_period = 5)

        reporter = CLIReporter(metric_columns=['loss', 'accuracy', 'training_iteration'])

        analysis = tune.run(
            trainable,
            resources_per_trial = resources_per_trial,
            config = config,
            num_samples = 1,
            scheduler=asha_scheduler,
            progress_reporter=reporter)

        best_trial = analysis.get_best_trial("loss", "min", "last")
        learning_rate = best_trial.config['lr']

        print('--------------------------------Start Simulations--------------------------------')
        # get test result of the trained model
        server = Server(arc(num_features=num_features, num_classes=2, seed = seed), info, train_prn = False, seed = seed, Z = Z, ret = True, prn = False,  metric = metric)
        trained_model = copy.deepcopy(server.model)
        trained_model.load_state_dict(torch.load(os.path.join(best_trial.checkpoint.value, 'checkpoint')))
        test_acc, n_yz = server.test_inference(trained_model)
        df = pd.DataFrame([{'accuracy': test_acc, 'DP Disp': DPDisparity(n_yz)}])

        # use the same hyperparameters for other seeds
        for seed in range(1, num_sim):
            print('--------------------------------Seed:' + str(seed) + '--------------------------------')
            result = run_dp(method = method, model = model, dataset = dataset, metric = metric, prn = False, seed = seed, learning_rate = learning_rate, **kwargs)
            df = df.append(pd.DataFrame([result]))
        df = df.reset_index(drop = True)
        acc_mean, dp_mean = df.mean()
        acc_std, dp_std = df.std()
        print("Result across %d simulations: " % num_sim)
        print("| Accuracy: %.4f(%.4f) | DP Disp: %.4f(%.4f)" % (acc_mean, acc_std, dp_mean, dp_std))
        return acc_mean, acc_std, dp_mean, dp_std

    elif method == 'fedfb':
        print('--------------------------------Hyperparameter selection--------------------------------')
        print('--------------------------------Seed:' + str(seed) + '--------------------------------')
        config = {'lr': tune.grid_search([.001, .005, .01]),
                'alpha': tune.grid_search([.001, .05, .08, .1, .2, .5, 1, 2])}

        def trainable(config): 
            return run_dp(method = method, model = model, dataset = dataset, metric = metric, prn = False, trial = True, seed = seed, learning_rate = config['lr'], alpha = config['alpha'], **kwargs)

        asha_scheduler = ASHAScheduler(
            time_attr = 'iteration',
            metric = 'disp',
            mode = 'min',
            grace_period = 5)

        reporter = CLIReporter(metric_columns=['loss', 'accuracy', 'iteration', 'disp'])

        analysis = tune.run(
            trainable,
            resources_per_trial = resources_per_trial,
            config = config,
            num_samples = 1,
            scheduler=asha_scheduler,
            progress_reporter=reporter)

        best_trial = analysis.get_best_trial("disp", "min", "last")
        params = best_trial.config
        learning_rate, alpha = params['lr'], params['alpha']

        print('--------------------------------Start Simulations--------------------------------')
        # get test result of the trained model
        server = Server(arc(num_features=num_features, num_classes=2, seed = seed), info, train_prn = False, seed = seed, Z = Z, ret = True, prn = False, metric = metric)
        trained_model = copy.deepcopy(server.model)
        trained_model.load_state_dict(torch.load(os.path.join(best_trial.checkpoint.value, 'checkpoint')))
        test_acc, n_yz = server.test_inference(trained_model)
        df = pd.DataFrame([{'accuracy': test_acc, 'DP Disp': DPDisparity(n_yz)}])

        # use the same hyperparameters for other seeds
        for seed in range(1, num_sim):
            print('--------------------------------Seed:' + str(seed) + '--------------------------------')
            result = run_dp(method = method, model = model, dataset = dataset, metric = metric, prn = False, seed = seed, learning_rate = learning_rate, alpha = alpha, **kwargs)
            df = df.append(pd.DataFrame([result]))
        df = df.reset_index(drop = True)
        acc_mean, dp_mean = df.mean()
        acc_std, dp_std = df.std()
        print("Result across %d simulations: " % num_sim)
        print("| Accuracy: %.4f(%.4f) | DP Disp: %.4f(%.4f)" % (acc_mean, acc_std, dp_mean, dp_std))
        return acc_mean, acc_std, dp_mean, dp_std

    elif method == 'cflfb':
        print('--------------------------------Hyperparameter selection--------------------------------')
        print('--------------------------------Seed:' + str(seed) + '--------------------------------')
        config = {'lr': tune.grid_search([.001, .005]),
                'alpha': tune.grid_search([.001, .05, .08, .1, .2]),
                'rounds': tune.grid_search([1,10])}

        def trainable(config): 
            return run_dp(method = method, model = model, dataset = dataset, metric = metric, prn = False, trial = True, seed = seed, learning_rate = config['lr'], alpha = config['alpha'], outer_rounds = config['rounds'], inner_epochs = 300//config['rounds'], **kwargs)

        asha_scheduler = ASHAScheduler(
            time_attr = 'iteration',
            metric = 'disp',
            mode = 'min',
            grace_period = 50)

        reporter = CLIReporter(metric_columns=['loss', 'accuracy', 'training_iteration', 'disp'])

        analysis = tune.run(
            trainable,
            resources_per_trial = resources_per_trial,
            config = config,
            num_samples = 1,
            scheduler=asha_scheduler,
            progress_reporter=reporter)

        best_trial = analysis.get_best_trial("disp", "min", "last")
        params = best_trial.config
        learning_rate, alpha, rounds = params['lr'], params['alpha'], params['rounds']
        print("The hyperparameter we select is | learning rate: %.4f | alpha: %.4f " % (learning_rate, alpha))

        print('--------------------------------Start Simulations--------------------------------')
        # get test result of the trained model
        server = Server(arc(num_features=num_features, num_classes=2, seed = seed), info, train_prn = False, seed = seed, Z = Z, ret = True, prn = False, metric = metric)
        trained_model = copy.deepcopy(server.model)
        trained_model.load_state_dict(torch.load(os.path.join(best_trial.checkpoint.value, 'checkpoint')))
        test_acc, n_yz = server.test_inference(trained_model)
        df = pd.DataFrame([{'accuracy': test_acc, 'DP Disp': DPDisparity(n_yz)}])

        # use the same hyperparameters for other seeds
        for seed in range(1, num_sim):
            print('--------------------------------Seed:' + str(seed) + '--------------------------------')
            result = run_dp(method = method, model = model, dataset = dataset, metric = metric, prn = False, seed = seed, learning_rate = learning_rate, alpha = alpha, outer_rounds = rounds, inner_epochs = 300//rounds, **kwargs)
            df = df.append(pd.DataFrame([result]))
        df = df.reset_index(drop = True)
        acc_mean, dp_mean = df.mean()
        acc_std, dp_std = df.std()
        print("Result across %d simulations: " % num_sim)
        print("| Accuracy: %.4f(%.4f) | DP Disp: %.4f(%.4f)" % (acc_mean, acc_std, dp_mean, dp_std))
        return acc_mean, acc_std, dp_mean, dp_std

    elif method == 'uflfb':
        print('--------------------------------Hyperparameter selection--------------------------------')
        print('--------------------------------Seed:' + str(seed) + '--------------------------------')
        num_clients = len(info[2])
        if num_clients <= 2:
            params_array = cartesian([[.001, .01, .1]]*num_clients).tolist()
            # params_array = cartesian([[.01]]*num_clients).tolist()
            def trainable(config): 
                return run_dp(method = method, model = model, dataset = dataset, metric = metric, prn = False, seed = seed, learning_rate = [0.005] * num_clients, alpha = config['alpha'], **kwargs)
        else:
            params_array = [.001, .002, .005, .01, .02, .05, .1, 1]
            def trainable(config): 
                return run_dp(method = method, model = model, dataset = dataset, metric = metric, prn = False, seed = seed, learning_rate = [0.005] * num_clients, alpha = [config['alpha']] * num_clients, **kwargs)
        config = {'alpha': tune.grid_search(params_array)}

        analysis = tune.run(
            trainable,
            resources_per_trial = resources_per_trial,
            config = config,
            num_samples = 1)

        params = analysis.get_best_config(metric = "DP Disp", mode = "min")
        alpha = params['alpha']
        df = analysis.results_df[['accuracy', 'DP Disp']]

        print('--------------------------------Start Simulations--------------------------------')
        # use the same hyperparameters for other seeds
        for seed in range(1, num_sim):
            print('--------------------------------Seed:' + str(seed) + '--------------------------------')
            if num_clients <= 2:
                result = run_dp(method = method, model = model, dataset = dataset, metric = metric, prn = False, seed = seed, learning_rate = [0.005] * num_clients, alpha = alpha, **kwargs)
            else:
                result = run_dp(method = method, model = model, dataset = dataset, metric = metric, prn = False, seed = seed, learning_rate = [0.005] * num_clients, alpha = [alpha] * num_clients, **kwargs)
            df = df.append(pd.DataFrame([result]))
        df = df.reset_index(drop = True)
        acc_mean, dp_mean = df.mean()
        acc_std, dp_std = df.std()
        print("Result across %d simulations: " % num_sim)
        print("| Accuracy: %.4f(%.4f) | DP Disp: %.4f(%.4f)" % (acc_mean, acc_std, dp_mean, dp_std))
        return acc_mean, acc_std, dp_mean, dp_std
    
    elif method == 'fflfb':
        print('--------------------------------Hyperparameter selection--------------------------------')
        print('--------------------------------Seed:' + str(seed) + '--------------------------------')
        num_clients = len(info[2])
        if num_clients <= 2:
            params_array = cartesian([[.001, .01, .1]]*num_clients).tolist()
            # params_array = cartesian([[.01]]*num_clients).tolist()
            def trainable(config): 
                return run_dp(method = method, model = model, dataset = dataset, metric = metric, prn = False, trial = True, seed = seed, learning_rate = 0.005, alpha = config['alpha'], **kwargs)
        else:
            params_array = [.001, .002, .005, .01, .02, .05, .1, 1]
            def trainable(config): 
                return run_dp(method = method, model = model, dataset = dataset, metric = metric, prn = False, trial = True, seed = seed, learning_rate = 0.005, alpha = [config['alpha']] * num_clients, **kwargs)
        config = {'alpha': tune.grid_search(params_array)}

        asha_scheduler = ASHAScheduler(
            time_attr = 'iteration',
            metric = 'disp',
            mode = 'min',
            grace_period = 5)

        reporter = CLIReporter(metric_columns=['loss', 'accuracy', 'iteration', 'disp'])

        analysis = tune.run(
            trainable,
            resources_per_trial = resources_per_trial,
            config = config,
            num_samples = 1,
            scheduler=asha_scheduler,
            progress_reporter=reporter)

        best_trial = analysis.get_best_trial("disp", "min", "last")
        params = best_trial.config
        alpha = params['alpha']

        print('--------------------------------Start Simulations--------------------------------')
        # get test result of the trained model
        server = Server(arc(num_features=num_features, num_classes=2, seed = seed), info, train_prn = False, seed = seed, Z = Z, ret = True, prn = False, metric = metric)
        trained_model = copy.deepcopy(server.model)
        trained_model.load_state_dict(torch.load(os.path.join(best_trial.checkpoint.value, 'checkpoint')))
        test_acc, n_yz = server.test_inference(trained_model)
        df = pd.DataFrame([{'accuracy': test_acc, 'DP Disp': DPDisparity(n_yz)}])

        # use the same hyperparameters for other seeds
        for seed in range(1, num_sim):
            print('--------------------------------Seed:' + str(seed) + '--------------------------------')
            if num_clients <= 2:
                result = run_dp(method = method, model = model, dataset = dataset, metric = metric, prn = False, seed = seed, learning_rate = 0.005, alpha = alpha, **kwargs)
            else:
                result = run_dp(method = method, model = model, dataset = dataset, metric = metric, prn = False, seed = seed, learning_rate = 0.005, alpha = [alpha] * num_clients, **kwargs)
            df = df.append(pd.DataFrame([result]))
        df = df.reset_index(drop = True)
        acc_mean, dp_mean = df.mean()
        acc_std, dp_std = df.std()
        print("Result across %d simulations: " % num_sim)
        print("| Accuracy: %.4f(%.4f) | DP Disp: %.4f(%.4f)" % (acc_mean, acc_std, dp_mean, dp_std))
        return acc_mean, acc_std, dp_mean, dp_std

    elif method == 'fairfed':
        print('--------------------------------Hyperparameter selection--------------------------------')
        print('--------------------------------Seed:' + str(seed) + '--------------------------------')
        config = {'lr': tune.grid_search([.001, .005,]),
                'alpha': tune.grid_search([.0001, .0005, .001, .005, .01, .01, 1]),
                'beta': tune.grid_search([1])}

        def trainable(config): 
            return run_dp(method = method, model = model, dataset = dataset, metric = metric, prn = False, trial = True, seed = seed, learning_rate = config['lr'], alpha = config['alpha'], beta = config['beta'], **kwargs)

        asha_scheduler = ASHAScheduler(
            time_attr = 'iteration',
            metric = 'disp',
            mode = 'min',
            grace_period = 5)

        reporter = CLIReporter(metric_columns=['loss', 'accuracy', 'iteration', 'disp'])

        analysis = tune.run(
            trainable,
            resources_per_trial = resources_per_trial,
            config = config,
            num_samples = 1,
            scheduler=asha_scheduler,
            progress_reporter=reporter)

        best_trial = analysis.get_best_trial("disp", "min", "last")
        params = best_trial.config
        learning_rate, alpha, beta = params['lr'], params['alpha'], params['beta']

        print('--------------------------------Start Simulations--------------------------------')
        # get test result of the trained model
        server = Server(arc(num_features=num_features, num_classes=2, seed = seed), info, train_prn = False, seed = seed, Z = Z, ret = True, prn = False, metric = metric)
        trained_model = copy.deepcopy(server.model)
        trained_model.load_state_dict(torch.load(os.path.join(best_trial.checkpoint.value, 'checkpoint')))
        test_acc, n_yz = server.test_inference(trained_model)
        df = pd.DataFrame([{'accuracy': test_acc, 'DP Disp': DPDisparity(n_yz)}])

        # use the same hyperparameters for other seeds
        for seed in range(1, num_sim):
            print('--------------------------------Seed:' + str(seed) + '--------------------------------')
            result = run_dp(method = method, model = model, dataset = dataset, metric = metric, prn = False, seed = seed, learning_rate = learning_rate, alpha = alpha, beta = beta, **kwargs)
            df = df.append(pd.DataFrame([result]))
        df = df.reset_index(drop = True)
        acc_mean, dp_mean = df.mean()
        acc_std, dp_std = df.std()
        print("Result across %d simulations: " % num_sim)
        print("| Accuracy: %.4f(%.4f) | DP Disp: %.4f(%.4f)" % (acc_mean, acc_std, dp_mean, dp_std))
        return acc_mean, acc_std, dp_mean, dp_std


    else: 
        Warning('Does not support this method!')
        exit(1)

def sim_dp_man(method, model, dataset, metric = "Demographic disparity", num_sim = 5, seed = 0, **kwargs):
    results = []
    for seed in range(num_sim):
        results.append(run_dp(method, model, dataset, metric = metric, prn = True, seed = seed, trial = False, **kwargs))
    df = pd.DataFrame(results)
    acc_mean, rp_mean = df.mean()
    acc_mean, dp_mean = df.mean()
    acc_std, dp_std = df.std()
    print("Result across %d simulations: " % num_sim)
    print("| Accuracy: %.4f(%.4f) | DP Disp: %.4f(%.4f)" % (acc_mean, acc_std, dp_mean, dp_std))
    return acc_mean, acc_std, dp_mean, dp_std