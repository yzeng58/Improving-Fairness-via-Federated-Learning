# load modules and dataset
from ray.tune.progress_reporter import CLIReporter
from RP_server import *
from RP_load_dataset import *
from ray import tune
from ray.tune.schedulers import ASHAScheduler
import pandas as pd

def run_rp(method, model, dataset, prn = True, seed = 123, trial = False, **kwargs):
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
    elif dataset == 'compas':
        Z, num_features, info = compas_z, compas_num_features, compas_info
    elif dataset == 'communities':
        Z, num_features, info = communities_z, communities_num_features, communities_info
    elif dataset == 'bank':
        Z, num_features, info = bank_z, bank_num_features, bank_info
    else:
        Warning('Does not support this dataset!')
        exit(1)

    # set up the server
    server = Server(arc(num_features=num_features, num_classes=2, seed = seed), info, train_prn = False, seed = seed, Z = Z, ret = True, prn = prn, trial = trial)

    # execute
    if method == 'fedavg':
        acc, va, rp, _ = server.FedAvg(**kwargs)
    elif method == 'fedfb':
        acc, va, rp, _ = server.FedFB(**kwargs)
    elif method == 'gifair':
        acc, va, rp, _ = server.GIFAIR(**kwargs)
    elif method == 'ditto':
        acc, va, rp, _ = server.Ditto(**kwargs)
    elif method == 'qffl':
        acc, va, rp, _ = server.qFFL(**kwargs)
    else:
        Warning('Does not support this method!')
        exit(1)

    if not trial: return {'accuracy': acc, 'Var(accuracy)': va, 'Representation disparity': rp}

def sim_rp(method, model, dataset, num_sim = 5, seed = 0, resources_per_trial = {'cpu':4}, **kwargs):
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
    elif dataset == 'compas':
        Z, num_features, info = compas_z, compas_num_features, compas_info
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
        # config = {'lr': tune.grid_search([.001, .002, .005, .01, .02])}
        config = {'lr': tune.grid_search([.001, .005, .01])}
        def trainable(config): 
            return run_rp(method = method, model = model, dataset = dataset, prn = False, trial = True, seed = seed, learning_rate = config['lr'], **kwargs)

        asha_scheduler = ASHAScheduler(
            time_attr = 'iteration',
            metric = 'loss',
            mode = 'min',
            grace_period = 5)

        reporter = CLIReporter(metric_columns=['loss', 'accuracy', 'training_iteration', 'va', 'rp'])

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
        server = Server(arc(num_features=num_features, num_classes=2, seed = seed), info, train_prn = False, seed = seed, Z = Z, ret = True, prn = False)
        trained_model = copy.deepcopy(server.model)
        trained_model.load_state_dict(torch.load(os.path.join(best_trial.checkpoint.value, 'checkpoint')))
        test_acc, acc_z, loss_z = server.test_inference(trained_model)
        df = pd.DataFrame([{'accuracy': test_acc, 'Var(accuracy)': accVariance(acc_z), 'Representation disparity': RepresentationDisparity(loss_z)}])

        # use the same hyperparameters for other seeds
        for seed in range(1, num_sim):
            print('--------------------------------Seed:' + str(seed) + '--------------------------------')
            result = run_rp(method = method, model = model, dataset = dataset, prn = False, seed = seed, learning_rate = learning_rate, **kwargs)
            df = df.append(pd.DataFrame([result]))
        df = df.reset_index(drop = True)
        acc_mean, va_mean, rp_mean = df.mean()
        acc_std, va_std, rp_std = df.std()
        print("Result across %d simulations: " % num_sim)
        print("| Accuracy: %.4f(%.4f) | Var(accuracy): %.4f(%.4f) | Representation disparity: %.4f(%.4f)" % (acc_mean, acc_std, va_mean, va_std, rp_mean, rp_std))
        return acc_mean, acc_std, va_mean, va_std, rp_mean, rp_std

    elif method == 'fedfb':
        print('--------------------------------Hyperparameter selection--------------------------------')
        print('--------------------------------Seed:' + str(seed) + '--------------------------------')
        config = {'lr': tune.grid_search([.001, .005, .01]),
                'alpha': tune.grid_search([.05, .1, .5, 1, 2, 5, 10])}

        def trainable(config): 
            return run_rp(method = method, model = model, dataset = dataset, prn = False, trial = True, seed = seed, learning_rate = config['lr'], alpha = config['alpha'])

        asha_scheduler = ASHAScheduler(
            time_attr = 'iteration',
            metric = 'rp',
            mode = 'min',
            grace_period = 5)

        reporter = CLIReporter(metric_columns=['loss', 'accuracy', 'iteration', 'va', 'rp'])

        analysis = tune.run(
            trainable,
            resources_per_trial = resources_per_trial,
            config = config,
            num_samples = 1,
            scheduler=asha_scheduler,
            progress_reporter=reporter)

        best_trial = analysis.get_best_trial("rp", "min", "last")
        params = best_trial.config
        learning_rate, alpha = params['lr'], params['alpha']

        print('--------------------------------Start Simulations--------------------------------')
        # get test result of the trained model
        server = Server(arc(num_features=num_features, num_classes=2, seed = seed), info, train_prn = False, seed = seed, Z = Z, ret = True, prn = False)
        trained_model = copy.deepcopy(server.model)
        trained_model.load_state_dict(torch.load(os.path.join(best_trial.checkpoint.value, 'checkpoint')))
        test_acc, a_z, loss_z = server.test_inference(trained_model)
        df = pd.DataFrame([{'accuracy': test_acc, 'Var(accuracy)': accVariance(a_z), 'Representation disparity': RepresentationDisparity(loss_z)}])

        # use the same hyperparameters for other seeds
        for seed in range(1, num_sim):
            print('--------------------------------Seed:' + str(seed) + '--------------------------------')
            result = run_rp(method = method, model = model, dataset = dataset, prn = False, seed = seed, learning_rate = learning_rate, alpha = alpha)
            df = df.append(pd.DataFrame([result]))
        df = df.reset_index(drop = True)
        acc_mean, va_mean, rp_mean = df.mean()
        acc_std, va_std, rp_std = df.std()
        print("Result across %d simulations: " % num_sim)
        print("| Accuracy: %.4f(%.4f) | Var(accuracy): %.4f(%.4f) | Representation disparity: %.4f(%.4f)" % (acc_mean, acc_std, va_mean, va_std, rp_mean, rp_std))
        return acc_mean, acc_std, va_mean, va_std, rp_mean, rp_std

    elif method == 'gifair': 
        print('--------------------------------Hyperparameter selection--------------------------------')
        print('--------------------------------Seed:' + str(seed) + '--------------------------------')
        lbd_max = np.array(list(map(len, info[2])))
        lbd_max = (lbd_max / lbd_max.sum()).max()
        config = {'lr': tune.grid_search([.001, .005, .01]),
                'alpha': tune.grid_search([.1*lbd_max, .2*lbd_max, .3*lbd_max, .4*lbd_max, .6*lbd_max, .8*lbd_max])}

        def trainable(config): 
            return run_rp(method = method, model = model, dataset = dataset, prn = False, trial = True, seed = seed, learning_rate = config['lr'], alpha = config['alpha'])

        asha_scheduler = ASHAScheduler(
            time_attr = 'iteration',
            metric = 'rp',
            mode = 'min',
            grace_period = 5)

        reporter = CLIReporter(metric_columns=['loss', 'accuracy', 'iteration', 'va', 'rp'])

        analysis = tune.run(
            trainable,
            resources_per_trial = resources_per_trial,
            config = config,
            num_samples = 1,
            scheduler=asha_scheduler,
            progress_reporter=reporter)

        best_trial = analysis.get_best_trial("rp", "min", "last")
        params = best_trial.config
        learning_rate, alpha = params['lr'], params['alpha']

        print('--------------------------------Start Simulations--------------------------------')
        # get test result of the trained model
        server = Server(arc(num_features=num_features, num_classes=2, seed = seed), info, train_prn = False, seed = seed, Z = Z, ret = True, prn = False)
        trained_model = copy.deepcopy(server.model)
        trained_model.load_state_dict(torch.load(os.path.join(best_trial.checkpoint.value, 'checkpoint')))
        test_acc, a_z, loss_z = server.test_inference(trained_model)
        df = pd.DataFrame([{'accuracy': test_acc, 'Var(accuracy)': accVariance(a_z), 'Representation disparity': RepresentationDisparity(loss_z)}])

        # use the same hyperparameters for other seeds
        for seed in range(1, num_sim):
            print('--------------------------------Seed:' + str(seed) + '--------------------------------')
            result = run_rp(method = method, model = model, dataset = dataset, prn = False, seed = seed, learning_rate = learning_rate, alpha = alpha)
            df = df.append(pd.DataFrame([result]))
        df = df.reset_index(drop = True)
        acc_mean, va_mean, rp_mean = df.mean()
        acc_std, va_std, rp_std = df.std()
        print("Result across %d simulations: " % num_sim)
        print("| Accuracy: %.4f(%.4f) | Var(accuracy): %.4f(%.4f) | Representation disparity: %.4f(%.4f)" % (acc_mean, acc_std, va_mean, va_std, rp_mean, rp_std))
        return acc_mean, acc_std, va_mean, va_std, rp_mean, rp_std

    elif method == 'qffl':
        print('--------------------------------Hyperparameter selection--------------------------------')
        print('--------------------------------Seed:' + str(seed) + '--------------------------------')
        config = {'lr': tune.grid_search([.001, .005, .01]),
                'q': tune.grid_search([0, .001, .01, .1, 1, 2, 5, 10])}

        def trainable(config): 
            return run_rp(method = method, model = model, dataset = dataset, prn = False, trial = True, seed = seed, learning_rate = config['lr'], q = config['q'])

        asha_scheduler = ASHAScheduler(
            time_attr = 'iteration',
            metric = 'rp',
            mode = 'min',
            grace_period = 5)

        reporter = CLIReporter(metric_columns=['loss', 'accuracy', 'iteration', 'va', 'rp'])

        analysis = tune.run(
            trainable,
            resources_per_trial = resources_per_trial,
            config = config,
            num_samples = 1,
            scheduler=asha_scheduler,
            progress_reporter=reporter)

        best_trial = analysis.get_best_trial("rp", "min", "last")
        params = best_trial.config
        learning_rate, q = params['lr'], params['q']

        print('--------------------------------Start Simulations--------------------------------')
        # get test result of the trained model
        server = Server(arc(num_features=num_features, num_classes=2, seed = seed), info, train_prn = False, seed = seed, Z = Z, ret = True, prn = False)
        trained_model = copy.deepcopy(server.model)
        trained_model.load_state_dict(torch.load(os.path.join(best_trial.checkpoint.value, 'checkpoint')))
        test_acc, a_z, loss_z = server.test_inference(trained_model)
        df = pd.DataFrame([{'accuracy': test_acc, 'Var(accuracy)': accVariance(a_z), 'Representation disparity': RepresentationDisparity(loss_z)}])

        # use the same hyperparameters for other seeds
        for seed in range(1, num_sim):
            print('--------------------------------Seed:' + str(seed) + '--------------------------------')
            result = run_rp(method = method, model = model, dataset = dataset, prn = False, seed = seed, learning_rate = learning_rate, q = q)
            df = df.append(pd.DataFrame([result]))
        df = df.reset_index(drop = True)
        acc_mean, va_mean, rp_mean = df.mean()
        acc_std, va_std, rp_std = df.std()
        print("Result across %d simulations: " % num_sim)
        print("| Accuracy: %.4f(%.4f) | Var(accuracy): %.4f(%.4f) | Representation disparity: %.4f(%.4f)" % (acc_mean, acc_std, va_mean, va_std, rp_mean, rp_std))
        return acc_mean, acc_std, va_mean, va_std, rp_mean, rp_std

    elif method == 'ditto':
        print('--------------------------------Hyperparameter selection--------------------------------')
        print('--------------------------------Seed:' + str(seed) + '--------------------------------')
        config = {'lr': tune.grid_search([.001, .005, .01]),
                'penalty': tune.grid_search([.01, .05, .1, .5, 1, 2, 5])}

        def trainable(config): 
            return run_rp(method = method, model = model, dataset = dataset, prn = False, trial = True, seed = seed, learning_rate = config['lr'], penalty = config['penalty'])

        asha_scheduler = ASHAScheduler(
            time_attr = 'iteration',
            metric = 'rp',
            mode = 'min',
            grace_period = 5)

        reporter = CLIReporter(metric_columns=['loss', 'accuracy', 'iteration', 'va', 'rp'])

        analysis = tune.run(
            trainable,
            resources_per_trial = resources_per_trial,
            config = config,
            num_samples = 1,
            scheduler=asha_scheduler,
            progress_reporter=reporter)

        best_trial = analysis.get_best_trial("rp", "min", "last")
        params = best_trial.config
        learning_rate, penalty = params['lr'], params['penalty']

        print('--------------------------------Start Simulations--------------------------------')
        # get test result of the trained model
        server = Server(arc(num_features=num_features, num_classes=2, seed = seed), info, train_prn = False, seed = seed, Z = Z, ret = True, prn = False)
        data_saved = torch.load(os.path.join(best_trial.checkpoint.value, 'checkpoint'))
        models_v = []
        for c in range(len(info[2])):
            models_v.append(copy.deepcopy(server.model))
            models_v[c].load_state_dict(data_saved[c])
        test_acc, a_z, loss_z = server.mtl_inference(models_v)
        df = pd.DataFrame([{'accuracy': test_acc, 'Var(accuracy)': accVariance(a_z), 'Representation disparity': RepresentationDisparity(loss_z)}])

        # use the same hyperparameters for other seeds
        for seed in range(1, num_sim):
            print('--------------------------------Seed:' + str(seed) + '--------------------------------')
            result = run_rp(method = method, model = model, dataset = dataset, prn = False, seed = seed, learning_rate = learning_rate, penalty = penalty)
            df = df.append(pd.DataFrame([result]))
        df = df.reset_index(drop = True)
        acc_mean, va_mean, rp_mean = df.mean()
        acc_std, va_std, rp_std = df.std()
        print("Result across %d simulations: " % num_sim)
        print("| Accuracy: %.4f(%.4f) | Var(accuracy): %.4f(%.4f) | Representation disparity: %.4f(%.4f)" % (acc_mean, acc_std, va_mean, va_std, rp_mean, rp_std))
        return acc_mean, acc_std, va_mean, va_std, rp_mean, rp_std

    else: 
        Warning('Does not support this method!')
        exit(1)

def sim_rp_man(method, model, dataset, num_sim = 5, seed = 0, **kwargs):
    results = []
    for seed in range(num_sim):
        results.append(run_rp(method, model, dataset, prn = True, seed = seed, trial = False, **kwargs))
    df = pd.DataFrame(results)
    acc_mean, va_mean, rp_mean = df.mean()
    acc_std, va_std, rp_std = df.std()
    print("Result across %d simulations: " % num_sim)
    print("| Accuracy: %.4f(%.4f) | Var(accuracy): %.4f(%.4f) | Representation disparity: %.4f(%.4f)" % (acc_mean, acc_std, va_mean, va_std, rp_mean, rp_std))
    return acc_mean, acc_std, va_mean, va_std, rp_mean, rp_std

