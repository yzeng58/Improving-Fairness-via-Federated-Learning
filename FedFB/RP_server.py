# the index of z is the same as the index of the client
import torch, copy, time, random, torch, os
import numpy as np

from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils import *
from ray import tune

################## MODEL SETTING ########################
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
os.environ['KMP_DUPLICATE_LIB_OK']='True'
#########################################################

class Server(object):
    def __init__(self, model, dataset_info, seed = 123, num_workers = 4, ret = False, 
                train_prn = False, batch_size = 128, print_every = 1, fraction_clients = 1, Z = 2, trial = False, prn = True):
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
        self.seed = seed
        self.num_workers = num_workers

        self.ret = ret
        self.prn = prn
        self.train_prn = False if ret else train_prn

        self.metric = "Representation Disparity"
        self.disparity = RepresentationDisparity

        self.batch_size = batch_size
        self.print_every = print_every
        self.fraction_clients = fraction_clients

        self.train_dataset, self.test_dataset, self.clients_idx = dataset_info
        self.num_clients = len(self.clients_idx)
        self.Z = Z

        self.trainloader, self.validloader = self.train_val(self.train_dataset, batch_size)

        self.trial = trial
    
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

    def FedAvg(self, num_rounds = 10, local_epochs = 30, learning_rate = 0.005, optimizer = "adam"):
        # set seed
        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.manual_seed(self.seed)

        # Training
        train_loss, train_accuracy = [], []
        start_time = time.time()
        weights = self.model.state_dict()
        
        for round_ in tqdm(range(num_rounds)):
            local_weights, local_losses = [], []
            if self.prn: print(f'\n | Global Training Round : {round_+1} |\n')

            self.model.train()

            for idx in range(self.num_clients):
                local_model = Client(dataset=self.train_dataset, idxs=self.clients_idx[idx], 
                            batch_size = self.batch_size, option = "unconstrained", seed = self.seed, prn = self.train_prn, Z = self.Z)

                w, loss = local_model.standard_update(
                                model=copy.deepcopy(self.model), global_round=round_, 
                                    learning_rate = learning_rate, local_epochs = local_epochs, 
                                    optimizer = optimizer)
                local_weights.append(copy.deepcopy(w))
                local_losses.append(copy.deepcopy(loss))

            # update global weights
            weights = average_weights(local_weights, self.clients_idx, list(range(self.num_clients)))
            self.model.load_state_dict(weights)

            loss_avg = sum(local_losses) / len(local_losses)
            train_loss.append(loss_avg)

            # Calculate avg training accuracy over all clients at every round
            list_acc = []
            # the number of samples which are assigned to class y and belong to the sensitive group z
            loss_z = np.zeros(self.num_clients)
            acc_z = np.zeros(self.num_clients)
            self.model.eval()
            for c in range(self.num_clients):
                local_model = Client(dataset=self.train_dataset, idxs=self.clients_idx[c], 
                            batch_size = self.batch_size, option = "unconstrained", seed = self.seed, prn = self.train_prn, Z = self.Z)
                # validation dataset inference
                acc_z[c], loss_z[c], acc_loss, fair_loss = local_model.inference(model = self.model) 
                list_acc.append(acc_z[c])
                    
                if self.prn: 
                    print("Client %d: accuracy loss: %.2f | fairness loss %.2f " % (
                            c+1, acc_loss, fair_loss))

            train_accuracy.append(sum(list_acc)/len(list_acc))

            # print global training loss after every 'i' rounds
            if self.prn:
                if (round_+1) % self.print_every == 0:
                    print(f' \nAvg Training Stats after {round_+1} global rounds:')
                    print("Training loss: %.2f | Validation accuracy: %.2f%% | Validation Representation Disparity: %.4f | Validation Accuracy Variance: %.4f" % (
                        np.mean(np.array(train_loss)), 
                        100*train_accuracy[-1], RepresentationDisparity(loss_z), accVariance(acc_z)))

            if self.trial:
                with tune.checkpoint_dir(round_) as checkpoint_dir:
                    path = os.path.join(checkpoint_dir, "checkpoint")
                    torch.save(self.model.state_dict(), path)
                    
                tune.report(loss = loss, accuracy = train_accuracy[-1], va = accVariance(acc_z), iteration = round_+1, rp = RepresentationDisparity(loss_z))

        # Test inference after completion of training
        test_acc, acc_z, loss_z = self.test_inference()
        rd = accVariance(acc_z)
        rp = RepresentationDisparity(loss_z)

        if self.prn:
            print(f' \n Results after {num_rounds} global rounds of training:')
            print("|---- Avg Train Accuracy: {:.2f}%".format(100*train_accuracy[-1]))
            print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))

            # Compute fairness metric
            print("|---- Test Accuracy Variance: {:.4f}".format(rd))

            print('\n Total Run Time: {0:0.4f} sec'.format(time.time()-start_time))

        if self.ret: return test_acc, rd, rp, self.model

    def FedFB(self, num_rounds = 10, local_epochs = 30, learning_rate = 0.005, optimizer = 'adam', alpha = 0.3):
        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.manual_seed(self.seed)

        # Training
        train_loss, train_accuracy = [], []
        start_time = time.time()
        weights = self.model.state_dict()

        # the number of samples whose label is y and sensitive attribute is z
        m_z, lbd = [], []
        for z in range(self.Z):
            m_z.append(len(self.clients_idx[z]))
            lbd.append((m_z[z])/len(self.train_dataset))

        for round_ in tqdm(range(num_rounds)):
            local_weights, local_losses = [], []
            if self.prn: print(f'\n | Global Training Round : {round_+1} |\n')

            self.model.train()

            for idx in range(self.num_clients):
                local_model = Client(dataset=self.train_dataset,
                                            idxs=self.clients_idx[idx], batch_size = self.batch_size, 
                                        option = "FB-Variant1", 
                                        seed = self.seed, prn = self.train_prn, Z = self.Z)

                w, loss = local_model.fb2_update(model=copy.deepcopy(self.model), global_round=round_, 
                                    learning_rate = learning_rate, local_epochs = local_epochs, 
                                    optimizer = optimizer, m_z = m_z, lbd = lbd, z = idx)
                local_weights.append(copy.deepcopy(w))
                local_losses.append(copy.deepcopy(loss))

            # update global weights
            nc = np.array(m_z) * np.array(lbd)
            weights = weighted_average_weights(local_weights, nc, sum(nc))
            self.model.load_state_dict(weights)

            loss_avg = sum(local_losses) / len(local_losses)
            train_loss.append(loss_avg)

            # Calculate avg training accuracy over all clients at every round
            list_acc = []
            # the number of samples which are assigned to class y and belong to the sensitive group z
            loss_z = np.zeros(self.Z)
            acc_z = np.zeros(self.Z)

            self.model.eval()
            for c in range(self.num_clients):
                local_model = Client(dataset=self.train_dataset,
                                            idxs=self.clients_idx[c], batch_size = self.batch_size, option = "FB-Variant1", 
                                            seed = self.seed, prn = self.train_prn, Z = self.Z)
                # validation dataset inference
                acc_z[c], loss_z[c], acc_loss, fair_loss = local_model.inference(model = self.model, train = True) 
                list_acc.append(acc_z[c])
                    
                if self.prn: print("Client %d: accuracy loss: %.2f | fairness loss %.2f" % (
                    c+1, acc_loss, fair_loss))

            for z in range(self.Z):
                if z == 0:
                    lbd[0] -= alpha / (round_ + 1) ** .5 * sum([(loss_z[z] - loss_z[0]) for z in range(self.Z)])
                    # lbd[0] = lbd[0].item()
                    # lbd[0] = max(0, min(lbd[0], 2*(m_z[z])/len(self.train_dataset)))
                else:
                    lbd[z] += alpha / (round_ + 1) ** .5 * (loss_z[z] - loss_z[0])
                    # lbd[z] = lbd[z].item()
                    # lbd[z] = max(0, min(lbd[z], 2*(m_z[z])/len(self.train_dataset)))
            
            train_accuracy.append(sum(list_acc)/len(list_acc))

            # print global training loss after every 'i' rounds
            if self.prn:
                if (round_+1) % self.print_every == 0:
                    print(f' \nAvg Training Stats after {round_+1} global rounds:')
                    print("Training loss: %.2f | Training accuracy: %.2f%% | Training Representation Disparity: %.4f | Training Accuracy Variance: %.4f " % (
                        np.mean(np.array(train_loss)), 
                        100*train_accuracy[-1], RepresentationDisparity(loss_z), accVariance(acc_z)))

            if self.trial:
                with tune.checkpoint_dir(round_) as checkpoint_dir:
                    path = os.path.join(checkpoint_dir, "checkpoint")
                    torch.save(self.model.state_dict(), path)
                    
                tune.report(loss = loss, accuracy = train_accuracy[-1], va = accVariance(acc_z), iteration = round_+1, rp = RepresentationDisparity(loss_z))

        # Test inference after completion of training
        test_acc, acc_z, loss_z = self.test_inference(self.model, self.test_dataset)
        rd = accVariance(acc_z)
        rp = RepresentationDisparity(loss_z)

        if self.prn:
            print(f' \n Results after {num_rounds} global rounds of training:')
            print("|---- Avg Train Accuracy: {:.2f}%".format(100*train_accuracy[-1]))
            print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))

            # Compute fairness metric
            print("|---- Test Accuracy Variance: {:.4f}".format(rd))
            print("|---- Test Representation Disparity: {:.4f}".format(rp))

            print('\n Total Run Time: {0:0.4f} sec'.format(time.time()-start_time))

        if self.ret: return test_acc, rd, rp, self.model

    def GIFAIR(self, num_rounds = 10, local_epochs = 30, learning_rate = 0.005, optimizer = 'adam', alpha = 0.3):
        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.manual_seed(self.seed)

        # Training
        train_loss, train_accuracy = [], []
        start_time = time.time()
        weights = self.model.state_dict()

        # the number of samples whose label is y and sensitive attribute is z
        m_z, lbd = [], []
        for z in range(self.Z):
            m_z.append((self.train_dataset.sen == z).sum())
            lbd.append((m_z[z])/len(self.train_dataset))

        for round_ in tqdm(range(num_rounds)):
            local_weights, local_losses = [], []
            if self.prn: print(f'\n | Global Training Round : {round_+1} |\n')

            self.model.train()

            for idx in range(self.num_clients):
                local_model = Client(dataset=self.train_dataset,
                                            idxs=self.clients_idx[idx], batch_size = self.batch_size, 
                                        option = "FB-Variant1", 
                                        seed = self.seed, prn = self.train_prn, Z = self.Z)

                w, loss = local_model.fb2_update(model=copy.deepcopy(self.model), global_round=round_, 
                                    learning_rate = learning_rate, local_epochs = local_epochs, 
                                    optimizer = optimizer, m_z = m_z, lbd = lbd, z = idx)
                local_weights.append(copy.deepcopy(w))
                local_losses.append(copy.deepcopy(loss))

            # update global weights
            nc = np.array(m_z) * np.array(lbd)
            weights = weighted_average_weights(local_weights, nc, sum(nc))
            self.model.load_state_dict(weights)

            loss_avg = sum(local_losses) / len(local_losses)
            train_loss.append(loss_avg)

            # Calculate avg training accuracy over all clients at every round
            list_acc = []
            # the number of samples which are assigned to class y and belong to the sensitive group z
            loss_z = np.zeros(self.Z)
            acc_z = np.zeros(self.Z)

            self.model.eval()
            for c in range(self.num_clients):
                local_model = Client(dataset=self.train_dataset,
                                            idxs=self.clients_idx[c], batch_size = self.batch_size, option = "FB-Variant1", 
                                            seed = self.seed, prn = self.train_prn, Z = self.Z)
                # validation dataset inference
                acc_z[c], loss_z[c], acc_loss, fair_loss = local_model.inference(model = self.model, train = True) 
                list_acc.append(acc_z[c])
                    
                if self.prn: print("Client %d: accuracy loss: %.2f | fairness loss %.2f" % (
                    c+1, acc_loss, fair_loss))

            z_idx = sorted(list(range(self.Z)), key = lambda z: loss_z[z])
            rk = np.arange(-self.Z + 1, self.Z + 1, 2)
            for i in range(self.Z):
                lbd[z_idx[i]] = (m_z[z])/len(self.train_dataset) + alpha * rk[i]
            
            train_accuracy.append(sum(list_acc)/len(list_acc))

            # print global training loss after every 'i' rounds
            if self.prn:
                if (round_+1) % self.print_every == 0:
                    print(f' \nAvg Training Stats after {round_+1} global rounds:')
                    print("Training loss: %.2f | Training accuracy: %.2f%% | Training Representation Disparity: %.4f | Training Accuracy Variance: %.4f " % (
                        np.mean(np.array(train_loss)), 
                        100*train_accuracy[-1], RepresentationDisparity(loss_z), accVariance(acc_z)))

            if self.trial:
                with tune.checkpoint_dir(round_) as checkpoint_dir:
                    path = os.path.join(checkpoint_dir, "checkpoint")
                    torch.save(self.model.state_dict(), path)
                    
                tune.report(loss = loss, accuracy = train_accuracy[-1], va = accVariance(acc_z), iteration = round_+1, rp = RepresentationDisparity(loss_z))

        # Test inference after completion of training
        test_acc, acc_z, loss_z = self.test_inference(self.model, self.test_dataset)
        rd = accVariance(acc_z)
        rp = RepresentationDisparity(loss_z)

        if self.prn:
            print(f' \n Results after {num_rounds} global rounds of training:')
            print("|---- Avg Train Accuracy: {:.2f}%".format(100*train_accuracy[-1]))
            print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))

            # Compute fairness metric
            print("|---- Test Accuracy Variance: {:.4f}".format(rd))
            print("|---- Test Representation Disparity: {:.4f}".format(rp))

            print('\n Total Run Time: {0:0.4f} sec'.format(time.time()-start_time))

        if self.ret: return test_acc, rd, rp, self.model

    def qFFL(self, num_rounds = 10, local_epochs = 30, learning_rate = 0.005, optimizer = 'adam', q = 0.3):
        # set seed
        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.manual_seed(self.seed)

        L = 1/learning_rate
        # Training
        train_loss, train_accuracy = [], []
        start_time = time.time()
        tem_model = copy.deepcopy(self.model)
        
        for round_ in tqdm(range(num_rounds)):
            local_losses = []
            if self.prn: print(f'\n | Global Training Round : {round_+1} |\n')

            self.model.train()

            sum_deltakt, sum_hkt = None, 0
            for idx in range(self.num_clients):
                local_model = Client(dataset=self.train_dataset, idxs=self.clients_idx[idx], 
                            batch_size = self.batch_size, option = "unconstrained", seed = self.seed, prn = self.train_prn, Z = self.Z)

                w, loss = local_model.standard_update(
                                model=copy.deepcopy(self.model), global_round=round_, 
                                    learning_rate = learning_rate, local_epochs = local_epochs, 
                                    optimizer = optimizer)
                tem_model.load_state_dict(w)
                deltakt, hkt = local_model.qffl_compute(model=self.model, 
                                     q = q, L = L, local_model = tem_model)

                if sum_deltakt == None:
                    sum_deltakt = deltakt
                else:
                    for key in deltakt:
                        sum_deltakt[key] += deltakt[key]
                sum_hkt += hkt
                local_losses.append(copy.deepcopy(loss))

            # update global weights
            weights = self.model.state_dict()
            for key in weights:
                weights[key] = weights[key] - sum_deltakt[key]/sum_hkt
            self.model.load_state_dict(weights)

            loss_avg = sum(local_losses) / len(local_losses)
            train_loss.append(loss_avg)

            # Calculate avg training accuracy over all clients at every round
            list_acc = []
            # the number of samples which are assigned to class y and belong to the sensitive group z
            loss_z = np.zeros(self.num_clients)
            acc_z = np.zeros(self.num_clients)
            self.model.eval()
            for c in range(self.num_clients):
                local_model = Client(dataset=self.train_dataset, idxs=self.clients_idx[c], 
                            batch_size = self.batch_size, option = "unconstrained", seed = self.seed, prn = self.train_prn, Z = self.Z)
                # validation dataset inference
                acc_z[c], loss_z[c], acc_loss, fair_loss = local_model.inference(model = self.model) 
                list_acc.append(acc_z[c])
                    
                if self.prn: 
                    print("Client %d: accuracy loss: %.2f | fairness loss %.2f " % (
                            c+1, acc_loss, fair_loss))

            train_accuracy.append(sum(list_acc)/len(list_acc))

            # print global training loss after every 'i' rounds
            if self.prn:
                if (round_+1) % self.print_every == 0:
                    print(f' \nAvg Training Stats after {round_+1} global rounds:')
                    print("Training loss: %.2f | Validation accuracy: %.2f%% | Validation Representation Disparity: %.4f | Validation Accuracy Variance: %.4f" % (
                        np.mean(np.array(train_loss)), 
                        100*train_accuracy[-1], RepresentationDisparity(loss_z), accVariance(acc_z)))

            if self.trial:
                with tune.checkpoint_dir(round_) as checkpoint_dir:
                    path = os.path.join(checkpoint_dir, "checkpoint")
                    torch.save(self.model.state_dict(), path)
                    
                tune.report(loss = loss, accuracy = train_accuracy[-1], va = accVariance(acc_z), iteration = round_+1, rp =  RepresentationDisparity(loss_z))

        # Test inference after completion of training
        test_acc, acc_z, loss_z = self.test_inference()
        rd = accVariance(acc_z)
        rp = RepresentationDisparity(loss_z)

        if self.prn:
            print(f' \n Results after {num_rounds} global rounds of training:')
            print("|---- Avg Train Accuracy: {:.2f}%".format(100*train_accuracy[-1]))
            print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))

            # Compute fairness metric
            print("|---- Test Accuracy Variance: {:.4f}".format(rd))
            print("|---- Test Representation Disparity: {:.4f}".format(rp))

            print('\n Total Run Time: {0:0.4f} sec'.format(time.time()-start_time))

        if self.ret: return test_acc, rd, rp, self.model

    def Ditto(self, num_rounds = 10, local_epochs = 30, learning_rate = 0.005, optimizer = 'adam', penalty = 0.3):
        # set seed
        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.manual_seed(self.seed)

        # Training
        train_loss, train_accuracy = [], []
        start_time = time.time()
        weights = self.model.state_dict()

        models_v = [copy.deepcopy(self.model) for _ in range(self.num_clients)]
        # models_w = copy.deepcopy(models_v)

        for round_ in tqdm(range(num_rounds)):
            local_weights, local_losses = [], []
            if self.prn: print(f'\n | Global Training Round : {round_+1} |\n')

            for idx in range(self.num_clients):
                models_v[idx].train()
                self.model.train()
                # models_w[idx].train()
                local_model = Client(dataset=self.train_dataset, idxs=self.clients_idx[idx], 
                            batch_size = self.batch_size, option = "unconstrained", seed = self.seed, prn = self.train_prn, Z = self.Z)

                w, loss = local_model.standard_update(
                                model=copy.deepcopy(self.model), global_round=round_, 
                                    learning_rate = learning_rate, local_epochs = local_epochs, 
                                    optimizer = optimizer)

                loss = local_model.ditto_update(
                                model=models_v[idx], global_model = copy.deepcopy(self.model), global_round=round_, 
                                    learning_rate = learning_rate, local_epochs = local_epochs, 
                                    optimizer = optimizer, penalty = penalty)
                # models[idx].load_state_dict(w)
                local_weights.append(copy.deepcopy(w))
                local_losses.append(copy.deepcopy(loss))

            # update global weights
            weights = average_weights(local_weights, self.clients_idx, list(range(self.num_clients)))
            self.model.load_state_dict(weights)

            loss_avg = sum(local_losses) / len(local_losses)
            train_loss.append(loss_avg)

            # Calculate avg training accuracy over all clients at every round
            list_acc = []
            # the number of samples which are assigned to class y and belong to the sensitive group z
            loss_z = np.zeros(self.num_clients)
            acc_z = np.zeros(self.num_clients)
            self.model.eval()
            for c in range(self.num_clients):
                local_model = Client(dataset=self.train_dataset, idxs=self.clients_idx[c], 
                            batch_size = self.batch_size, option = "unconstrained", seed = self.seed, prn = self.train_prn, Z = self.Z)
                # validation dataset inference
                acc_z[c], loss_z[c], acc_loss, fair_loss = local_model.inference(model = models_v[c]) 
                list_acc.append(acc_z[c])
                    
                if self.prn: 
                    print("Client %d: accuracy loss: %.2f | fairness loss %.2f " % (
                            c+1, acc_loss, fair_loss))

            train_accuracy.append(sum(list_acc)/len(list_acc))

            # print global training loss after every 'i' rounds
            if self.prn:
                if (round_+1) % self.print_every == 0:
                    print(f' \nAvg Training Stats after {round_+1} global rounds:')
                    print("Training loss: %.2f | Validation accuracy: %.2f%% | Validation Representation Disparity: %.4f | Validation Accuracy Variance: %.4f" % (
                        np.mean(np.array(train_loss)), 
                        100*train_accuracy[-1], RepresentationDisparity(loss_z), accVariance(acc_z)))

            if self.trial:
                with tune.checkpoint_dir(round_) as checkpoint_dir:
                    path = os.path.join(checkpoint_dir, "checkpoint")
                    torch.save([m.state_dict() for m in models_v], path)
                    
                tune.report(loss = loss, accuracy = train_accuracy[-1], va = accVariance(acc_z), iteration = round_+1, rp = RepresentationDisparity(loss_z))

        # Test inference after completion of training
        test_acc, acc_z, loss_z = self.mtl_inference(models = models_v)
        rd = accVariance(acc_z)
        rp = RepresentationDisparity(loss_z)

        if self.prn:
            print(f' \n Results after {num_rounds} global rounds of training:')
            print("|---- Avg Train Accuracy: {:.2f}%".format(100*train_accuracy[-1]))
            print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))

            # Compute fairness metric
            print("|---- Test Accuracy Variance: {:.4f}".format(rd))
            print("|---- Test Representation Disparity: {:.4f}".format(rp))

            print('\n Total Run Time: {0:0.4f} sec'.format(time.time()-start_time))

        if self.ret: return test_acc, rd, rp, models_v 

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
        acc_z = np.zeros(self.num_clients)
        
        testloader = DataLoader(test_dataset, batch_size=self.batch_size,
                                shuffle=False)

        loss_z = np.zeros(self.Z)
        for _, (features, labels, sensitive) in enumerate(testloader):
            features = features.to(DEVICE)
            labels =  labels.to(DEVICE).type(torch.LongTensor)
            # Inference
            outputs, logits = model(features)

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            bool_correct = torch.eq(pred_labels, labels)
            correct += torch.sum(bool_correct).item()
            total += len(labels)

            group_boolean_idx = {}
            
            for z in range(self.num_clients):
                acc_z[z] += torch.sum((sensitive == z) & bool_correct).item()  / (self.test_dataset.sen == z).sum()
                group_boolean_idx[z] = (sensitive == z)
                batch_loss,_,_ = loss_func('unconstrained', logits[group_boolean_idx[z]], labels[group_boolean_idx[z]], outputs[group_boolean_idx[z]], sensitive[group_boolean_idx[z]], 0)
                loss_z[z] += batch_loss / (self.test_dataset.sen == z).sum()

        accuracy = correct/total

        return accuracy, acc_z, loss_z

    def mtl_inference(self, models, test_dataset = None):
        """ 
        Returns the test accuracy and fairness level.
        """
        # set seed
        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.manual_seed(self.seed)

        if test_dataset == None: test_dataset = self.test_dataset

        for model in models:
            model.eval()

        total, correct = 0.0, 0.0
        acc_z = np.zeros(self.num_clients)
        
        loss_z = np.zeros(self.Z)
        for z in range(self.Z):
            testloader = DataLoader(DatasetSplit(test_dataset, np.where(test_dataset.sen == z)[0]), batch_size=self.batch_size,
                                    shuffle=False)
            for _, (features, labels, sensitive) in enumerate(testloader):
                features = features.to(DEVICE)
                labels =  labels.to(DEVICE).type(torch.LongTensor)
                # Inference
                outputs, logits = models[z](features)

                # Prediction
                _, pred_labels = torch.max(outputs, 1)
                pred_labels = pred_labels.view(-1)
                bool_correct = torch.eq(pred_labels, labels)
                correct += torch.sum(bool_correct).item()
                total += len(labels)

                group_boolean_idx = {}
                
                for z in range(self.num_clients):
                    acc_z[z] += torch.sum((sensitive == z) & bool_correct).item()  / (self.test_dataset.sen == z).sum()
                    group_boolean_idx[z] = (sensitive == z)
                    batch_loss,_,_ = loss_func('unconstrained', logits[group_boolean_idx[z]], labels[group_boolean_idx[z]], outputs[group_boolean_idx[z]], sensitive[group_boolean_idx[z]], 0)
                    loss_z[z] += batch_loss / (self.test_dataset.sen == z).sum()

        accuracy = correct/total

        return accuracy, acc_z, loss_z

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

    def standard_update(self, model, global_round, learning_rate, local_epochs, optimizer): 
        # Set mode to train model
        model.train()
        epoch_loss = []

        # set seed
        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.manual_seed(self.seed)

        # Set optimizer for the local updates
        if optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate,
                                        ) # momentum=0.5
        elif optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                        weight_decay=1e-4)
        for i in range(local_epochs):
            batch_loss = []
            for batch_idx, (features, labels, sensitive) in enumerate(self.trainloader):
                features, labels = features.to(DEVICE), labels.to(DEVICE).type(torch.LongTensor)
                sensitive = sensitive.to(DEVICE)
                # we need to set the gradients to zero before starting to do backpropragation 
                # because PyTorch accumulates the gradients on subsequent backward passes. 
                # This is convenient while training RNNs
                
                probas, logits = model(features)
                loss, _, _ = loss_func(self.option, logits, labels, probas, sensitive, self.penalty)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if self.prn and (100. * batch_idx / len(self.trainloader)) % 50 == 0:
                    print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tBatch Loss: {:.6f}'.format(
                        global_round + 1, i, batch_idx * len(features),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        # weight, loss
        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def fb2_update(self, model, global_round, learning_rate, local_epochs, optimizer, lbd, m_z, z):
        # Set mode to train model
        model.train()
        epoch_loss = []
        v0 = lbd[z] * sum(m_z)/m_z[z]

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

                v = torch.ones(len(labels)).type(torch.DoubleTensor) * v0
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

        # weight, loss
        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def qffl_compute(self, model, q, L, local_model): 
        # Set mode to train model
        model.train()

        # set seed
        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.manual_seed(self.seed)

        features, labels = torch.tensor(self.train_dataset.x).to(DEVICE), torch.tensor(self.train_dataset.y).to(DEVICE).type(torch.LongTensor)
        _, logits = model(features)

        deltawkt = copy.deepcopy(model.state_dict())
        deltakt = copy.deepcopy(deltawkt)

        deltawkt_norm2 = 0.
        fk = F.cross_entropy(logits, labels, reduction = 'mean')
        for key in deltawkt:
            deltawkt[key] = L*(deltawkt[key] - local_model.state_dict()[key])
            deltawkt_norm2 += torch.norm(deltawkt[key])**2
            deltakt[key] = deltawkt[key] * fk**q
        hkt = q*fk**(q-1) * deltawkt_norm2 + L * fk**q
        return deltakt, hkt

    def ditto_update(self, model, global_model, global_round, learning_rate, local_epochs, optimizer, penalty): 
        # Set mode to train model
        model.train()
        epoch_loss = []

        # set seed
        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.manual_seed(self.seed)

        # Set optimizer for the local updates
        if optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate,
                                        ) # momentum=0.5
        elif optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                        weight_decay=1e-4)
        for i in range(local_epochs):
            batch_loss = []
            for batch_idx, (features, labels, sensitive) in enumerate(self.trainloader):
                features, labels = features.to(DEVICE), labels.to(DEVICE).type(torch.LongTensor)
                sensitive = sensitive.to(DEVICE)
                # we need to set the gradients to zero before starting to do backpropragation 
                # because PyTorch accumulates the gradients on subsequent backward passes. 
                # This is convenient while training RNNs
                
                probas, logits = model(features)
                loss = mtl_loss(logits, labels, penalty, global_model, model)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if self.prn and (100. * batch_idx / len(self.trainloader)) % 50 == 0:
                    print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tBatch Loss: {:.6f}'.format(
                        global_round + 1, i, batch_idx * len(features),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        # weight, loss
        return sum(epoch_loss) / len(epoch_loss)

    def inference(self, model, train = False):
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
        
        dataset = self.validloader if not train else self.trainloader
        for _, (features, labels, sensitive) in enumerate(dataset):
            features, labels = features.to(DEVICE), labels.to(DEVICE).type(torch.LongTensor)
            sensitive = sensitive.to(DEVICE).type(torch.LongTensor)
            
            # Inference
            outputs, logits = model(features)

            # Prediction
            
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            bool_correct = torch.eq(pred_labels, labels)
            correct += torch.sum(bool_correct).item()
            total += len(labels)
            num_batch += 1
            
            batch_loss, batch_acc_loss, batch_fair_loss = loss_func(self.option, logits, 
                                                        labels, outputs, sensitive, self.penalty)
            loss, acc_loss, fair_loss = (loss + batch_loss.item(), 
                                         acc_loss + batch_acc_loss.item(), 
                                         fair_loss + batch_fair_loss.item())
        accuracy = correct/total
        return accuracy, loss/total, acc_loss / num_batch, fair_loss / num_batch
