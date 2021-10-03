import torch, copy, time, random, warnings, os
import numpy as np

from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import *
from ray import tune
import torch.nn as nn

################## MODEL SETTING ########################
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
os.environ['KMP_DUPLICATE_LIB_OK']='True'
#########################################################

def laplace(x, μ, b):
    return 1 / (2 * b) * np.exp(-np.abs(x - μ) / b)

class Server(object):
    def __init__(self, model, dataset_info, seed = 123, num_workers = 4, ret = False, 
                train_prn = False, metric = "Demographic disparity", 
                batch_size = 128, print_every = 1, fraction_clients = 1, Z = 2, prn = True, trial = False, ε = 1):
        """
        Server execution.

        Parameters
        ----------
        model: torch.nn.Module object.

        dataset_info: a list of three objects.
            - train_dataset: Dataset object.
            - test_dataset: Dataset object.
            - clients_idx: a list of lists, with each sublist contains the indexs of the training samples in one client.
                    the length of the list is the number of clients.

        seed: random seed.

        num_workers: number of workers.

        ret: boolean value. If true, return the accuracy and fairness measure and print nothing; else print the log and return None.

        train_prn: boolean value. If true, print the batch loss in local epochs.

        metric: three options, "Risk Difference", "pRule", "Demographic disparity".

        batch_size: a positive integer.

        print_every: a positive integer. eg. print_every = 1 -> print the information of that global round every 1 round.

        fraction_clients: float from 0 to 1. The fraction of clients chose to update the weights in each round.
        """

        self.model = model
        if torch.cuda.device_count()>1:
            self.model = nn.DataParallel(self.model)
        self.model.to(DEVICE)

        self.seed = seed
        self.num_workers = num_workers

        self.ret = ret
        self.prn = prn
        self.train_prn = False if ret else train_prn

        self.metric = metric
        if metric == "Risk Difference":
            self.disparity = riskDifference
        elif metric == "pRule":
            self.disparity = pRule
        elif metric == "Demographic disparity":
            self.disparity = DPDisparity
        else:
            warnings.warn("Warning message: metric " + metric + " is not supported! Use the default metric Demographic disparity. ")
            self.disparity = DPDisparity
            self.metric = "Demographic disparity"

        self.batch_size = batch_size
        self.print_every = print_every
        self.fraction_clients = fraction_clients

        self.train_dataset, self.test_dataset, self.clients_idx = dataset_info
        self.num_clients = len(self.clients_idx)
        self.Z = Z

        self.trial = trial
        self.ε = ε
        self.trainloader, self.validloader = self.train_val(self.train_dataset, batch_size)
    
    def train_val(self, dataset, batch_size, idxs_train_full = None, split = False):
        """
        Returns train, validation for a given local training dataset
        and user indexes.
        """
        torch.manual_seed(self.seed)
        
        # split indexes for train, validation (90, 10)
        if idxs_train_full == None: idxs_train_full = np.arange(len(dataset))
        idxs_train = idxs_train_full[:int(0.9*len(idxs_train_full))]
        idxs_val = idxs_train_full[int(0.9*len(idxs_train_full)):]
    
        trainloader = DataLoader(DatasetSplit(dataset, idxs_train),
                                    batch_size=batch_size, shuffle=True)

        if split:
            validloader = {}
            for sen in range(self.Z):
                sen_idx = np.where(dataset.sen[idxs_val] == sen)[0]
                validloader[sen] = DataLoader(DatasetSplit(dataset, idxs_val[sen_idx]),
                                        batch_size=max(int(len(idxs_val)/10),10), shuffle=False)
        else:
            validloader = DataLoader(DatasetSplit(dataset, idxs_val),
                                     batch_size=max(int(len(idxs_val)/10),10), shuffle=False)
        return trainloader, validloader

    def FedFB(self, num_rounds = 10, local_epochs = 30, learning_rate = 0.005, optimizer = 'adam', alpha = 0.3):
        # only support 2 groups
        if self.Z == 2:
            # new algorithm for demographic parity, add weights directly, signed gradient-based algorithm
            # set seed
            np.random.seed(self.seed)
            random.seed(self.seed)
            torch.manual_seed(self.seed)

            # Training
            train_loss, train_accuracy = [], []
            start_time = time.time()
            weights = self.model.state_dict()

            # the number of samples whose label is y and sensitive attribute is z
            m_yz, lbd = {}, {}
            for y in [0,1]:
                for z in range(self.Z):
                    m_yz[(y,z)] = ((self.train_dataset.y == y) & (self.train_dataset.sen == z)).sum()

            for y in [0,1]:
                for z in range(self.Z):
                    lbd[(y,z)] = m_yz[(y,z)]/(m_yz[(0,z)] + m_yz[(1,z)])

            for round_ in tqdm(range(num_rounds)):
                local_weights, local_losses, nc = [], [], []
                if self.prn: print(f'\n | Global Training Round : {round_+1} |\n')

                self.model.train()

                for idx in range(self.num_clients):
                    local_model = Client(dataset=self.train_dataset,
                                                idxs=self.clients_idx[idx], batch_size = self.batch_size, 
                                            option = "FB-Variant1", 
                                            seed = self.seed, prn = self.train_prn, Z = self.Z)

                    w, loss, nc_ = local_model.fb_update(
                                    model=copy.deepcopy(self.model), global_round=round_, 
                                        learning_rate = learning_rate / (round_+1), local_epochs = local_epochs, 
                                        optimizer = optimizer, lbd = lbd, m_yz = m_yz, ε = self.ε)
                    nc.append(nc_)
                    local_weights.append(copy.deepcopy(w))
                    local_losses.append(copy.deepcopy(loss))

                # update global weights
                weights = weighted_average_weights(local_weights, nc, sum(nc))
                self.model.load_state_dict(weights)

                loss_avg = sum(local_losses) / len(local_losses)
                train_loss.append(loss_avg)

                # Calculate avg training accuracy over all clients at every round
                list_acc = []
                # the number of samples which are assigned to class y and belong to the sensitive group z
                n_yz, loss_yz = {}, {}
                for y in [0,1]:
                    for z in range(self.Z):
                        n_yz[(y,z)] = 0
                        loss_yz[(y,z)] = 0

                self.model.eval()
                for c in range(self.num_clients):
                    local_model = Client(dataset=self.train_dataset,
                                                idxs=self.clients_idx[c], batch_size = self.batch_size, option = "FB-Variant1", 
                                                seed = self.seed, prn = self.train_prn, Z = self.Z)
                    # validation dataset inference
                    acc, loss, n_yz_c, acc_loss, fair_loss, loss_yz_c = local_model.inference(model = self.model, ε = self.ε) 
                    list_acc.append(acc)
                    
                    for yz in n_yz:
                        n_yz[yz] += n_yz_c[yz]
                        loss_yz[yz] += loss_yz_c[yz]
                        
                    if self.prn: print("Client %d: accuracy loss: %.2f | fairness loss %.2f | %s = %.2f" % (
                        c+1, acc_loss, fair_loss, self.metric, self.disparity(n_yz_c)))
                    
                # update the lambda according to the paper -> see Section A.1 of FairBatch
                # works well! The real batch size would be slightly different from the setting
                for y, z in loss_yz:
                    loss_yz[(y,z)] = loss_yz[(y,z)]/(m_yz[(0,z)] + m_yz[(1,z)])

                y0_diff = loss_yz[(0,0)] - loss_yz[(0,1)]
                y1_diff = loss_yz[(1,0)] - loss_yz[(1,1)]
                if y0_diff > y1_diff:
                    lbd[(0,0)] -= alpha / (round_+1)
                    lbd[(0,0)] = min(max(0, lbd[(0,0)]), 1)
                    lbd[(1,0)] = 1 - lbd[(0,0)]
                    lbd[(0,1)] += alpha / (round_+1)
                    lbd[(0,1)] = min(max(0, lbd[(0,1)]), 1)
                    lbd[(1,1)] = 1 - lbd[(0,1)]
                else:
                    lbd[(0,0)] += alpha / (round_+1)
                    lbd[(0,0)] = min(max(0, lbd[(0,0)]), 1)
                    lbd[(0,1)] = 1 - lbd[(0,0)]
                    lbd[(1,0)] -= alpha / (round_+1)
                    lbd[(1,0)] = min(max(0, lbd[(1,0)]), 1)
                    lbd[(1,1)] = 1 - lbd[(1,0)]

                train_accuracy.append(sum(list_acc)/len(list_acc))

                # print global training loss after every 'i' rounds
                if self.prn:
                    if (round_+1) % self.print_every == 0:
                        print(f' \nAvg Training Stats after {round_+1} global rounds:')
                        print("Training loss: %.2f | Training accuracy: %.2f%% | Training %s: %.4f" % (
                            np.mean(np.array(train_loss)), 
                            100*train_accuracy[-1], self.metric, self.disparity(n_yz)))

                if self.trial:
                    with tune.checkpoint_dir(round_) as checkpoint_dir:
                        path = os.path.join(checkpoint_dir, "checkpoint")
                        torch.save(self.model.state_dict(), path)
                        
                    tune.report(loss = loss, accuracy = train_accuracy[-1], disp = self.disparity(n_yz), iteration = round_+1)

            # Test inference after completion of training
            test_acc, n_yz = self.test_inference(self.model, self.test_dataset)
            rd = self.disparity(n_yz)

            if self.prn:
                print(f' \n Results after {num_rounds} global rounds of training:')
                print("|---- Avg Train Accuracy: {:.2f}%".format(100*train_accuracy[-1]))
                print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))

                # Compute fairness metric
                print("|---- Test "+ self.metric+": {:.4f}".format(rd))

                print('\n Total Run Time: {0:0.4f} sec'.format(time.time()-start_time))

            if self.ret: return test_acc, rd, self.model

        # support more than 2 groups
        else:
            # new algorithm for demographic parity, add weights directly, signed gradient-based algorithm
            # set seed
            np.random.seed(self.seed)
            random.seed(self.seed)
            torch.manual_seed(self.seed)

            # Training
            train_loss, train_accuracy = [], []
            start_time = time.time()
            weights = self.model.state_dict()

            # the number of samples whose label is y and sensitive attribute is z
            m_yz, lbd = {}, {}
            for y in [0,1]:
                for z in range(self.Z):
                    m_yz[(y,z)] = ((self.train_dataset.y == y) & (self.train_dataset.sen == z)).sum()

            for y in [0,1]:
                for z in range(self.Z):
                    lbd[(y,z)] = (m_yz[(1,z)] + m_yz[(0,z)])/len(self.train_dataset)

            for round_ in tqdm(range(num_rounds)):
                local_weights, local_losses, nc = [], [], []
                if self.prn: print(f'\n | Global Training Round : {round_+1} |\n')

                self.model.train()

                for idx in range(self.num_clients):
                    local_model = Client(dataset=self.train_dataset,
                                                idxs=self.clients_idx[idx], batch_size = self.batch_size, 
                                            option = "FB-Variant1", 
                                            seed = self.seed, prn = self.train_prn, Z = self.Z)

                    w, loss, nc_ = local_model.fb2_update(
                                    model=copy.deepcopy(self.model), global_round=round_, 
                                        learning_rate = learning_rate, local_epochs = local_epochs, 
                                        optimizer = optimizer, m_yz = m_yz, lbd = lbd, ε = self.ε)
                    nc.append(nc_)
                    local_weights.append(copy.deepcopy(w))
                    local_losses.append(copy.deepcopy(loss))

                # update global weights
                weights = weighted_average_weights(local_weights, nc, sum(nc))
                self.model.load_state_dict(weights)

                loss_avg = sum(local_losses) / len(local_losses)
                train_loss.append(loss_avg)

                # Calculate avg training accuracy over all clients at every round
                list_acc = []
                # the number of samples which are assigned to class y and belong to the sensitive group z
                n_yz, loss_yz = {}, {}
                for y in [0,1]:
                    for z in range(self.Z):
                        n_yz[(y,z)] = 0
                        loss_yz[(y,z)] = 0

                self.model.eval()
                for c in range(self.num_clients):
                    local_model = Client(dataset=self.train_dataset,
                                                idxs=self.clients_idx[c], batch_size = self.batch_size, option = "FB-Variant1", 
                                                seed = self.seed, prn = self.train_prn, Z = self.Z)
                    # validation dataset inference
                    acc, loss, n_yz_c, acc_loss, fair_loss, loss_yz_c = local_model.inference(model = self.model, train = True, ε = self.ε) 
                    list_acc.append(acc)
                    
                    for yz in n_yz:
                        n_yz[yz] += n_yz_c[yz]
                        loss_yz[yz] += loss_yz_c[yz]
                        
                    if self.prn: print("Client %d: accuracy loss: %.2f | fairness loss %.2f | %s = %.2f" % (
                        c+1, acc_loss, fair_loss, self.metric, self.disparity(n_yz_c)))
                    
                for y, z in loss_yz:
                    loss_yz[(y,z)] = loss_yz[(y,z)]/(m_yz[(0,z)] + m_yz[(1,z)])

                for z in range(self.Z):
                    if z == 0:
                        lbd[(0,z)] -= alpha / (round_ + 1) ** .5 * sum([(loss_yz[(0,0)] + loss_yz[(1,0)] - loss_yz[(0,z)] - loss_yz[(1,z)]) for z in range(self.Z)])
                        lbd[(0,z)] = lbd[(0,z)].item()
                        lbd[(0,z)] = max(0, min(lbd[(0,z)], 2*(m_yz[(1,0)]+m_yz[(0,0)])/len(self.train_dataset)))
                        lbd[(1,z)] = 2*(m_yz[(1,0)]+m_yz[(0,0)])/len(self.train_dataset) - lbd[(0,z)]
                    else:
                        lbd[(0,z)] += alpha / (round_ + 1) ** .5 * (loss_yz[(0,0)] + loss_yz[(1,0)] - loss_yz[(0,z)] - loss_yz[(1,z)])
                        lbd[(0,z)] = lbd[(0,z)].item()
                        lbd[(0,z)] = max(0, min(lbd[(0,z)], 2*(m_yz[(1,0)]+m_yz[(0,0)])/len(self.train_dataset)))
                        lbd[(1,z)] = 2*(m_yz[(1,0)]+m_yz[(0,0)])/len(self.train_dataset) - lbd[(0,z)]
                
                train_accuracy.append(sum(list_acc)/len(list_acc))

                # print global training loss after every 'i' rounds
                if self.prn:
                    if (round_+1) % self.print_every == 0:
                        print(f' \nAvg Training Stats after {round_+1} global rounds:')
                        print("Training loss: %.2f | Training accuracy: %.2f%% | Training %s: %.4f" % (
                            np.mean(np.array(train_loss)), 
                            100*train_accuracy[-1], self.metric, self.disparity(n_yz)))

                if self.trial:
                    with tune.checkpoint_dir(round_) as checkpoint_dir:
                        path = os.path.join(checkpoint_dir, "checkpoint")
                        torch.save(self.model.state_dict(), path)
                        
                    tune.report(loss = loss, accuracy = train_accuracy[-1], disp = self.disparity(n_yz), iteration = round_+1)


            # Test inference after completion of training
            test_acc, n_yz = self.test_inference(self.model, self.test_dataset)
            rd = self.disparity(n_yz)

            if self.prn:
                print(f' \n Results after {num_rounds} global rounds of training:')
                print("|---- Avg Train Accuracy: {:.2f}%".format(100*train_accuracy[-1]))
                print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))

                # Compute fairness metric
                print("|---- Test "+ self.metric+": {:.4f}".format(rd))

                print('\n Total Run Time: {0:0.4f} sec'.format(time.time()-start_time))

            if self.ret: return test_acc, rd, self.model

    def FFLFB(self, num_rounds = 10, local_epochs = 30, learning_rate = 0.005, optimizer = 'adam', alpha = (0.3,0.3,0.3)):
        # new algorithm for demographic parity, add weights directly, signed gradient-based algorithm
        # set seed
        # set seed
        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.manual_seed(self.seed)

        # Training
        train_loss, train_accuracy = [], []
        start_time = time.time()
        weights = self.model.state_dict()

        lbd, m_yz, nc = [None for _ in range(self.num_clients)], [None for _ in range(self.num_clients)], [None for _ in range(self.num_clients)]

        for round_ in tqdm(range(num_rounds)):
            local_weights, local_losses = [], []
            if self.prn: print(f'\n | Global Training Round : {round_+1} |\n')

            self.model.train()
            # m = max(1, int(self.fraction_clients * self.num_clients)) # the number of clients to be chosen in each round_
            # idxs_users = np.random.choice(range(self.num_clients), m, replace=False)

            for idx in range(self.num_clients):
                local_model = Client(dataset=self.train_dataset, idxs=self.clients_idx[idx], 
                            batch_size = self.batch_size, option = "FB-Variant1", seed = self.seed, prn = self.train_prn, Z = self.Z)

                w, loss, nc_, lbd_, m_yz_ = local_model.local_fb(
                                model=copy.deepcopy(self.model), 
                                    learning_rate = learning_rate, local_epochs = local_epochs, 
                                    optimizer = optimizer, alpha = alpha[idx], lbd = lbd[idx], m_yz = m_yz[idx], ε = self.ε)
                lbd[idx], m_yz[idx], nc[idx] = lbd_, m_yz_, nc_
                local_weights.append(copy.deepcopy(w))
                local_losses.append(copy.deepcopy(loss))

            # update global weights
            weights = weighted_average_weights(local_weights, nc, sum(nc))
            self.model.load_state_dict(weights)

            loss_avg = sum(local_losses) / len(local_losses)
            train_loss.append(loss_avg)

            # Calculate avg training accuracy over all clients at every round
            list_acc = []
            # the number of samples which are assigned to class y and belong to the sensitive group z
            n_yz, loss_yz = {}, {}
            for y in [0,1]:
                for z in range(self.Z):
                    n_yz[(y,z)] = 0
                    loss_yz[(y,z)] = 0

            self.model.eval()
            for c in range(self.num_clients):
                local_model = Client(dataset=self.train_dataset,
                                            idxs=self.clients_idx[c], batch_size = self.batch_size, option = "FB-Variant1", 
                                            seed = self.seed, prn = self.train_prn, Z = self.Z)
                # validation dataset inference
                acc, loss, n_yz_c, acc_loss, fair_loss, loss_yz_c = local_model.inference(model = self.model, ε = self.ε) 
                list_acc.append(acc)
                
                for yz in n_yz:
                    n_yz[yz] += n_yz_c[yz]
                    loss_yz[yz] += loss_yz_c[yz]
                    
                if self.prn: print("Client %d: accuracy loss: %.2f | fairness loss %.2f | %s = %.2f" % (
                    c+1, acc_loss, fair_loss, self.metric, self.disparity(n_yz_c)))

            train_accuracy.append(sum(list_acc)/len(list_acc))

            # print global training loss after every 'i' rounds
            if self.prn:
                if (round_+1) % self.print_every == 0:
                    print(f' \nAvg Training Stats after {round_+1} global rounds:')
                    print("Training loss: %.2f | Validation accuracy: %.2f%% | Validation %s: %.4f" % (
                        np.mean(np.array(train_loss)), 
                        100*train_accuracy[-1], self.metric, self.disparity(n_yz)))

            if self.trial:
                with tune.checkpoint_dir(round_) as checkpoint_dir:
                    path = os.path.join(checkpoint_dir, "checkpoint")
                    torch.save(self.model.state_dict(), path)
                    
                tune.report(loss = loss, accuracy = train_accuracy[-1], disp = self.disparity(n_yz), iteration = round_+1)  

        # Test inference after completion of training
        test_acc, n_yz= self.test_inference()
        rd = self.disparity(n_yz)

        if self.prn:
            print(f' \n Results after {num_rounds} global rounds of training:')
            print("|---- Avg Train Accuracy: {:.2f}%".format(100*train_accuracy[-1]))
            print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))

            # Compute fairness metric
            print("|---- Test "+ self.metric+": {:.4f}".format(rd))

            print('\n Total Run Time: {0:0.4f} sec'.format(time.time()-start_time))

        if self.ret: return test_acc, rd, self.model

    def test_inference(self, model = None, test_dataset = None):

        """ 
        Returns the test accuracy and fairness level.
        """
        # set seed
        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.manual_seed(self.seed)

        if model == None: model = self.model
        if test_dataset == None: test_dataset = self.test_dataset

        model.eval()
        total, correct = 0.0, 0.0
        n_yz = {}
        for y in [0,1]:
            for z in range(self.Z):
                n_yz[(y,z)] = 0
        
        testloader = DataLoader(test_dataset, batch_size=self.batch_size,
                                shuffle=False)

        for _, (features, labels, sensitive) in enumerate(testloader):
            features = features.to(DEVICE)
            labels =  labels.to(DEVICE).type(torch.LongTensor)
            # Inference
            outputs, _ = model(features)

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            bool_correct = torch.eq(pred_labels, labels)
            correct += torch.sum(bool_correct).item()
            total += len(labels)
            
            for y,z in n_yz:
                n_yz[(y,z)] += torch.sum((sensitive == z) & (pred_labels == y)).item()  

        accuracy = correct/total

        return accuracy, n_yz

class Client(object):
    def __init__(self, dataset, idxs, batch_size, option, seed = 0, prn = True, penalty = 500, Z = 2):
        self.seed = seed 
        self.dataset = dataset
        self.idxs = idxs
        self.option = option
        self.prn = prn
        self.Z = Z
        self.trainloader, self.validloader = self.train_val(dataset, list(idxs), batch_size)
        self.penalty = penalty
        self.disparity = DPDisparity

    def train_val(self, dataset, idxs, batch_size):
        """
        Returns train, validation for a given local training dataset
        and user indexes.
        """
        torch.manual_seed(self.seed)
        
        # split indexes for train, validation (90, 10)
        idxs_train = idxs[:int(0.9*len(idxs))]
        idxs_val = idxs[int(0.9*len(idxs)):len(idxs)]

        self.train_dataset = DatasetSplit(dataset, idxs_train)
        self.test_dataset = DatasetSplit(dataset, idxs_val)

        trainloader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)

        validloader = DataLoader(self.test_dataset,
                                     batch_size=max(int(len(idxs_val)/10),10), shuffle=False)
        return trainloader, validloader

    def fb_update(self, model, global_round, learning_rate, local_epochs, optimizer, lbd, m_yz, ε):
        ####### Update ##########
        # Set mode to train model
        model.train()
        epoch_loss = []
        nc = 0

        # set seed
        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.manual_seed(self.seed)

        # Set optimizer for the local updates
        if optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate,
                                        momentum=0.5) # 
        elif optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                        weight_decay=1e-4)
        for i in range(local_epochs):
            batch_loss = []
            for batch_idx, (features, labels, sensitive) in enumerate(self.trainloader):
                features, labels = features.to(DEVICE), labels.type(torch.LongTensor).to(DEVICE)
                sensitive = sensitive.to(DEVICE)
                _, logits = model(features)

                logits = logits.to(DEVICE)
                v = torch.randn(len(labels)).type(torch.DoubleTensor).to(DEVICE)
                
                group_idx = {}
                
                for y, z in lbd:
                    group_idx[(y,z)] = torch.where((labels == y) & (sensitive == z))[0]
                    v[group_idx[(y,z)]] = lbd[(y,z)] * sum([m_yz[(y,z)] for z in range(self.Z)]) / m_yz[(y,z)]
                    nc += v[group_idx[(y,z)]].sum().item()

                # print(logits)
                loss = weighted_loss(logits, labels, v)
                # if global_round == 1: print(loss)

                optimizer.zero_grad()
                if not np.isnan(loss.item()): loss.backward()
                optimizer.step()

                if self.prn and (100. * batch_idx / len(self.trainloader)) % 50 == 0:
                    print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tBatch Loss: {:.6f}'.format(
                        global_round + 1, i, batch_idx * len(features),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        # weight, loss
        # ensure ε-differential privacy
        private_parameters = copy.deepcopy(model.state_dict())
        num_params = len(model.state_dict())
        each_ε = ε / (num_params + self.Z * 2)

        for key in model.state_dict(): 
            private_parameters[key] = model.state_dict()[key] + np.random.laplace(loc = 0, scale = 1/each_ε)
        model.load_state_dict(private_parameters)
        return model.state_dict(), sum(epoch_loss) / len(epoch_loss), nc

    def fb2_update(self, model, global_round, learning_rate, local_epochs, optimizer, lbd, m_yz, ε):
        # Set mode to train model
        model.train()
        epoch_loss = []
        nc = 0

        # set seed
        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.manual_seed(self.seed)

        # Set optimizer for the local updates
        if optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate,
                                        momentum=0.5) # 
        elif optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                        weight_decay=1e-4)
        for i in range(local_epochs):
            batch_loss = []
            for batch_idx, (features, labels, sensitive) in enumerate(self.trainloader):
                features, labels = features.to(DEVICE), labels.to(DEVICE).type(torch.LongTensor)
                sensitive = sensitive.to(DEVICE)
                _, logits = model(features)

                v = torch.ones(len(labels)).type(torch.DoubleTensor)
                
                group_idx = {}
                for y, z in lbd:
                    group_idx[(y,z)] = torch.where((labels == y) & (sensitive == z))[0]
                    v[group_idx[(y,z)]] = lbd[(y,z)] / (m_yz[(1,z)] + m_yz[(0,z)])
                    nc += v[group_idx[(y,z)]].sum().item()

                loss = weighted_loss(logits, labels, v, False)

                optimizer.zero_grad()
                if not np.isnan(loss.item()): loss.backward()
                optimizer.step()

                if self.prn and (100. * batch_idx / len(self.trainloader)) % 50 == 0:
                    print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tBatch Loss: {:.6f}'.format(
                        global_round + 1, i, batch_idx * len(features),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        # ensure ε-differential privacy
        private_parameters = copy.deepcopy(model.state_dict())
        num_params = len(model.state_dict())
        each_ε = ε / (num_params + self.Z * 2)

        for key in model.state_dict(): 
            private_parameters[key] = model.state_dict()[key] + np.random.laplace(loc = 0, scale = 1/each_ε)
        model.load_state_dict(private_parameters)
        return model.state_dict(), sum(epoch_loss) / len(epoch_loss), nc

    def local_fb(self, model, learning_rate, local_epochs, optimizer, alpha, lbd = None, m_yz = None, ε = 1):
        if self.Z == 2:
            # Set mode to train model
            epoch_loss = []
            nc = 0

            # set seed
            np.random.seed(self.seed)
            random.seed(self.seed)
            torch.manual_seed(self.seed)

            # Set optimizer for the local updates
            if optimizer == 'sgd':
                optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate,
                                            momentum=0.5) # 
            elif optimizer == 'adam':
                optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                            weight_decay=1e-4)
            
            if lbd == None:
                m_yz, lbd = {}, {}
                for y in [0,1]:
                    for z in range(self.Z):
                        m_yz[(y,z)] = ((self.dataset.y == y) & (self.dataset.sen == z)).sum()

                for y in [0,1]:
                    for z in range(self.Z):
                        lbd[(y,z)] = m_yz[(y,z)]/(m_yz[(0,z)] + m_yz[(1,z)])

            for epoch in range(local_epochs):
                model.train()
                batch_loss = []
                for _, (features, labels, sensitive) in enumerate(self.trainloader):
                    features, labels = features.to(DEVICE), labels.to(DEVICE).type(torch.LongTensor)
                    sensitive = sensitive.to(DEVICE)
                    _, logits = model(features)

                    v = torch.ones(len(labels)).type(torch.DoubleTensor)
                    
                    group_idx = {}
                    for y, z in lbd:
                        group_idx[(y,z)] = torch.where((labels == y) & (sensitive == z))[0]
                        v[group_idx[(y,z)]] = lbd[(y,z)] / (m_yz[(1,z)] + m_yz[(0,z)])
                        nc += v[group_idx[(y,z)]].sum().item()

                    loss = weighted_loss(logits, labels, v, False)

                    optimizer.zero_grad()
                    if not np.isnan(loss.item()): loss.backward()
                    optimizer.step()
                    batch_loss.append(loss.item())
                epoch_loss.append(sum(batch_loss)/len(batch_loss))

                model.eval()
                # validation dataset inference
                _, _, _, _, _, loss_yz = self.inference(model = model, train = True) 

                for y, z in loss_yz:
                    loss_yz[(y,z)] = loss_yz[(y,z)]/(m_yz[(0,z)] + m_yz[(1,z)])

                
                    y0_diff = loss_yz[(0,0)] - loss_yz[(0,1)]
                    y1_diff = loss_yz[(1,0)] - loss_yz[(1,1)]
                    if y0_diff > y1_diff:
                        lbd[(0,0)] -= alpha / (epoch+1)
                        lbd[(0,0)] = min(max(0, lbd[(0,0)]), 1)
                        lbd[(1,0)] = 1 - lbd[(0,0)]
                        lbd[(0,1)] += alpha / (epoch+1)
                        lbd[(0,1)] = min(max(0, lbd[(0,1)]), 1)
                        lbd[(1,1)] = 1 - lbd[(0,1)]
                    else:
                        lbd[(0,0)] += alpha / (epoch+1)
                        lbd[(0,0)] = min(max(0, lbd[(0,0)]), 1)
                        lbd[(0,1)] = 1 - lbd[(0,0)]
                        lbd[(1,0)] -= alpha / (epoch+1)
                        lbd[(1,0)] = min(max(0, lbd[(1,0)]), 1)
                        lbd[(1,1)] = 1 - lbd[(1,0)]

        else:
            epoch_loss = []
            nc = 0

            # set seed
            np.random.seed(self.seed)
            random.seed(self.seed)
            torch.manual_seed(self.seed)

            # Set optimizer for the local updates
            if optimizer == 'sgd':
                optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate,
                                            momentum=0.5) # 
            elif optimizer == 'adam':
                optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                            weight_decay=1e-4)

            if lbd == None:
                m_yz, lbd = {}, {}
                for y in [0,1]:
                    for z in range(self.Z):
                        m_yz[(y,z)] = ((self.dataset.y == y) & (self.dataset.sen == z)).sum()

                for y in [0,1]:
                    for z in range(self.Z):
                        lbd[(y,z)] = m_yz[(y,z)]/(m_yz[(0,z)] + m_yz[(1,z)])

            for i in range(local_epochs):
                batch_loss = []
                for batch_idx, (features, labels, sensitive) in enumerate(self.trainloader):
                    features, labels = features.to(DEVICE), labels.to(DEVICE).type(torch.LongTensor)
                    sensitive = sensitive.to(DEVICE)
                    _, logits = model(features)

                    v = torch.ones(len(labels)).type(torch.DoubleTensor)
                    
                    group_idx = {}
                    for y, z in lbd:
                        group_idx[(y,z)] = torch.where((labels == y) & (sensitive == z))[0]
                        v[group_idx[(y,z)]] = lbd[(y,z)] / (m_yz[(1,z)] + m_yz[(0,z)])
                        nc += v[group_idx[(y,z)]].sum().item()

                    loss = weighted_loss(logits, labels, v, False)

                    optimizer.zero_grad()
                    if not np.isnan(loss.item()): loss.backward()
                    optimizer.step()

                    if self.prn and (100. * batch_idx / len(self.trainloader)) % 50 == 0:
                        print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tBatch Loss: {:.6f}'.format(
                            global_round + 1, i, batch_idx * len(features),
                            len(self.trainloader.dataset),
                            100. * batch_idx / len(self.trainloader), loss.item()))
                    batch_loss.append(loss.item())
                epoch_loss.append(sum(batch_loss)/len(batch_loss))

            model.eval()
            # validation dataset inference
            _, _, _, _, _, loss_yz = self.inference(model = model, train = True) 

            for y, z in loss_yz:
                loss_yz[(y,z)] = loss_yz[(y,z)]/(m_yz[(0,z)] + m_yz[(1,z)])

            for z in range(self.Z):
                if z == 0:
                    lbd[(0,z)] -= alpha ** .5 * sum([(loss_yz[(0,0)] + loss_yz[(1,0)] - loss_yz[(0,z)] - loss_yz[(1,z)]) for z in range(self.Z)])
                    lbd[(0,z)] = lbd[(0,z)].item()
                    lbd[(0,z)] = max(0, min(lbd[(0,z)], 2*(m_yz[(1,0)]+m_yz[(0,0)])/len(self.train_dataset)))
                    lbd[(1,z)] = 2*(m_yz[(1,0)]+m_yz[(0,0)])/len(self.train_dataset) - lbd[(0,z)]
                else:
                    lbd[(0,z)] += alpha ** .5 * (loss_yz[(0,0)] + loss_yz[(1,0)] - loss_yz[(0,z)] - loss_yz[(1,z)])
                    lbd[(0,z)] = lbd[(0,z)].item()
                    lbd[(0,z)] = max(0, min(lbd[(0,z)], 2*(m_yz[(1,0)]+m_yz[(0,0)])/len(self.train_dataset)))
                    lbd[(1,z)] = 2*(m_yz[(1,0)]+m_yz[(0,0)])/len(self.train_dataset) - lbd[(0,z)]

        # ensure ε-differential privacy
        private_parameters = copy.deepcopy(model.state_dict())
        num_params = len(model.state_dict())
        each_ε = ε / num_params

        for key in model.state_dict(): 
            private_parameters[key] = model.state_dict()[key] + np.random.laplace(loc = 0, scale = 1/each_ε)
        model.load_state_dict(private_parameters)
        return model.state_dict(), sum(epoch_loss) / len(epoch_loss), nc, lbd, m_yz

    def inference(self, model, train = False, ε = 1):
        """ 
        Returns the inference accuracy, 
                                loss, 
                                N(sensitive group, pos), 
                                N(non-sensitive group, pos), 
                                N(sensitive group),
                                N(non-sensitive group),
                                acc_loss,
                                fair_loss
        """

        model.eval()
        loss, total, correct, fair_loss, acc_loss, num_batch = 0.0, 0.0, 0.0, 0.0, 0.0, 0
        n_yz, loss_yz = {}, {}
        for y in [0,1]:
            for z in range(self.Z):
                loss_yz[(y,z)] = 0
                n_yz[(y,z)] = 0
        
        dataset = self.validloader if not train else self.trainloader
        for _, (features, labels, sensitive) in enumerate(dataset):
            features, labels = features.to(DEVICE), labels.type(torch.LongTensor).to(DEVICE)
            sensitive = sensitive.type(torch.LongTensor).to(DEVICE)
            
            # Inference
            outputs, logits = model(features)
            outputs, logits = outputs.to(DEVICE), logits.to(DEVICE)

            # Prediction
            
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1).to(DEVICE)
            bool_correct = torch.eq(pred_labels, labels)
            correct += torch.sum(bool_correct).item()
            total += len(labels)
            num_batch += 1

            group_boolean_idx = {}
            
            for yz in n_yz:
                group_boolean_idx[yz] = (labels == yz[0]) & (sensitive == yz[1])
                n_yz[yz] += torch.sum((pred_labels == yz[0]) & (sensitive == yz[1])).item()     
                
                if self.option in["FairBatch", "FB-Variant1"]:
                # the objective function have no lagrangian term

                    loss_yz_,_,_ = loss_func("FB_inference", logits[group_boolean_idx[yz]].to(DEVICE), 
                                                    labels[group_boolean_idx[yz]].to(DEVICE), 
                                         outputs[group_boolean_idx[yz]].to(DEVICE), sensitive[group_boolean_idx[yz]].to(DEVICE), 
                                         self.penalty)
                    loss_yz[yz] += loss_yz_
            
            batch_loss, batch_acc_loss, batch_fair_loss = loss_func(self.option, logits, 
                                                        labels, outputs, sensitive, self.penalty)
            loss, acc_loss, fair_loss = (loss + batch_loss.item(), 
                                         acc_loss + batch_acc_loss.item(), 
                                         fair_loss + batch_fair_loss.item())
        accuracy = correct/total
        if self.option in ["FairBatch", "FB-Variant1"]:
            private_parameters = copy.deepcopy(model.state_dict())
            num_params = len(model.state_dict())
            each_ε = ε / (num_params + self.Z * 2)

            for yz in loss_yz:
                loss_yz[yz] += loss_yz[yz] + np.random.laplace(loc = 0, scale = 1/each_ε)
            return accuracy, loss, n_yz, acc_loss / num_batch, fair_loss / num_batch, loss_yz
        else:
            return accuracy, loss, n_yz, acc_loss / num_batch, fair_loss / num_batch, None
