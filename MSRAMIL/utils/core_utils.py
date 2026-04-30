import numpy as np
import pandas as pd
import torch

from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.metrics import auc as calc_auc
import json
from sklearn import preprocessing


import numpy as np
import torch
from utils.utils import *
import matplotlib.pyplot as plt
import sklearn.metrics as metrics

import os
from datasets.dataset_generic import save_splits
from models.model_MSRAMIL import MSRAMIL, MSRAMIL_C
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import auc as calc_auc
import json
from sklearn import preprocessing
import numpy as np
from sklearn.metrics import confusion_matrix
from utils.utils import calculate_error


def calculate_TP(y, y_pred):
    tp = 0
    for i, j in zip(y, y_pred):
        if i == j == 1:
            tp += 1
    return tp
def calculate_TN(y, y_pred):
    tn = 0
    for i, j in zip(y, y_pred):
        if i == j == 0:
            tn += 1
    return tn
def calculate_FP(y, y_pred):
    fp = 0
    for i, j in zip(y, y_pred):
        if i == 0 and j == 1:
            fp += 1
    return fp
def calculate_FN(y, y_pred):
    fn = 0
    for i, j in zip(y, y_pred):
        if i == 1 and j == 0:
            fn += 1
    return fn

def calculate_precision(y, y_pred):
    tp = calculate_TP(y, y_pred)
    fp = calculate_FP(y, y_pred)
    return (tp + 1e-9) / (tp + fp + 1e-9)

def calculate_recall(y, y_pred):
    tp = calculate_TP(y, y_pred)
    fn = calculate_FN(y, y_pred)
    return (tp + 1e-9) / (tp + fn + 1e-9)

def calculate_F1(y, y_pred):
    p = calculate_precision(y, y_pred)
    r = calculate_recall(y, y_pred)
    return 2*p*r / (p + r + 1e-9)

def calculate_specificity(y, y_pred):
    fp = calculate_FP(y, y_pred)
    tn = calculate_TN(y, y_pred)
    return tn / (tn + fp + 1e-9)

def plot_matrix(label_roc, pred_label, save_path1):
    C = confusion_matrix(label_roc, pred_label)
    plt.matshow(C, cmap=plt.cm.Blues)  # 根据最下面的图按自己需求更改颜色
    for i in range(len(C)):
        for j in range(len(C)):
            plt.annotate(C[j, i], xy=(i, j), horizontalalignment='center', verticalalignment='center')
    # plt.tick_params(labelsize=15) # 设置左边和上面的label类别如0,1,2,3,4的字体大小。
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(save_path1)

def plot_ROC(labels, preds, savepath):

    fpr1, tpr1, threshold1 = metrics.roc_curve(labels, preds)

    roc_auc1 = metrics.auc(fpr1, tpr1)
    plt.figure()
    lw = 2
    plt.figure(figsize=(10, 10))
    plt.plot(fpr1, tpr1, color='darkorange',
             lw=lw, label='AUC = %0.3f' % roc_auc1)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('1 - Specificity')
    plt.ylabel('Sensitivity')
    plt.legend(loc="lower right")
    plt.savefig(savepath)  # 保存文件
    return fpr1, tpr1

def plot_allROC(tprs, mean_fpr, aucs, savepath, fpr_all, tpr_all):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='grey',
            label='Chance', alpha=.8)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)

    ax.plot(mean_fpr, mean_tpr, color='royalblue',
            label=r'Mean ROC (AUC = %0.3f $\pm$ %0.3f)' % (mean_auc, std_auc),
            lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='royalblue', alpha=.2,
                    label=r'$\pm$ 1 std. dev.')
    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05])
    ax.legend(loc="lower right")
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')

    plt.show()
    plt.savefig(savepath)



def bootstrap_auc(y_true, y_pred, n_bootstraps=2000, rng_seed=42):
    n_bootstraps = n_bootstraps
    rng_seed = rng_seed
    bootstrapped_scores = []

    rng = np.random.RandomState(rng_seed)
    for i in range(n_bootstraps):
        indices = rng.randint(len(y_pred), size=len(y_pred))
        try:
            score = roc_auc_score(y_true[indices], y_pred[indices])
            bootstrapped_scores.append(score)
        except ValueError:
            pass
    bootstrapped_scores = np.array(bootstrapped_scores)

    print("AUROC: {:0.3f}".format(roc_auc_score(y_true, y_pred)))
    print("Confidence interval for the AUROC score: [{:0.3f} - {:0.3}]".format(
        np.percentile(bootstrapped_scores, (2.5, 97.5))[0], np.percentile(bootstrapped_scores, (2.5, 97.5))[1]))

    return roc_auc_score(y_true, y_pred), np.percentile(bootstrapped_scores, (2.5, 97.5))

class Accuracy_Logger(object):
    """Accuracy logger"""
    def __init__(self, n_classes):
        super(Accuracy_Logger, self).__init__()
        self.n_classes = n_classes
        self.initialize()

    def initialize(self):
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]
    
    def log(self, Y_hat, Y):
        Y_hat = int(Y_hat)
        Y = int(Y)
        self.data[Y]["count"] += 1
        self.data[Y]["correct"] += (Y_hat == Y)
    
    def log_batch(self, Y_hat, Y):
        Y_hat = np.array(Y_hat).astype(int)
        Y = np.array(Y).astype(int)
        for label_class in np.unique(Y):
            cls_mask = Y == label_class
            self.data[label_class]["count"] += cls_mask.sum()
            self.data[label_class]["correct"] += (Y_hat[cls_mask] == Y[cls_mask]).sum()
    
    def get_summary(self, c):
        count = self.data[c]["count"] 
        correct = self.data[c]["correct"]
        
        if count == 0: 
            acc = None
        else:
            acc = float(correct) / count
        
        return acc, correct, count

class EarlyStopping:

    def __init__(self, patience=20, stop_epoch=50, verbose=False):

        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_error_min = np.Inf

    def __call__(self, epoch, val_error, model, ckpt_name = 'checkpoint.pt'):

        score = -val_error

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_error, model, ckpt_name)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_error, model, ckpt_name)
            self.counter = 0


    def save_checkpoint(self, val_error, model, ckpt_name):
        '''Saves model when validation loss decrease.'''

        if self.verbose:
            print(f'Validation loss decreased ({self.val_error_min:.6f} --> {val_error:.6f}).  Saving model ...')
        torch.save(model.state_dict(), ckpt_name)
        self.val_error_min = val_error


def train(datasets, cur, args):
    '''
    cur = fold
    '''
    """   
        train for a single fold
    """
    print('\nTraining Fold {}!'.format(cur))
    writer_dir = os.path.join(args.results_dir, str(cur))
    results_dir = args.results_dir

    if not os.path.isdir(writer_dir):
        os.mkdir(writer_dir)

    if args.log_data:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(writer_dir, flush_secs=15)

    else:
        writer = None

    print('\nInit train/val/test splits...', end=' ')
    train_split, val_split, test_split = datasets
    save_splits(datasets, ['train', 'val', 'test'], os.path.join(args.results_dir, 'splits_{}.csv'.format(cur)))

    print('Done!')
    print("Training on {} samples".format(len(train_split)))
    print("Validating on {} samples".format(len(val_split)))
    print("Testing on {} samples".format(len(test_split)))

    print('\nInit loss function...', end=' ')
    if args.bag_loss == 'svm':
        from topk.svm import SmoothTop1SVM
        loss_fn = SmoothTop1SVM(n_classes=args.n_classes)
        if device.type == 'cuda':
            loss_fn = loss_fn.cuda()
    else:
        loss_fn = nn.CrossEntropyLoss()

        if device.type == 'cuda':
            loss_fn = loss_fn.cuda()

    print('Done!')

    print('\nInit Model...', end=' ')
    model_dict = {"dropout": args.drop_out, 'n_classes': args.n_classes}

    if args.model_size is not None and args.model_type != 'mil':
        model_dict.update({"size_arg": args.model_size})

    if args.model_type in ['MSRAMIL', 'clam_mb']:
        if args.subtyping:
            model_dict.update({'subtyping': True})

        if args.B > 0:
            model_dict.update({'k_sample': args.B})

        if args.inst_loss == 'svm':
            from topk.svm import SmoothTop1SVM
            instance_loss_fn = SmoothTop1SVM(n_classes=2)
            if device.type == 'cuda':
                instance_loss_fn = instance_loss_fn.cuda()
        else:

            instance_loss_fn = nn.CrossEntropyLoss()
        if args.model_type == 'MSRAMIL':
            if args.use_cli:
                model = MSRAMIL_C(**model_dict, instance_loss_fn=instance_loss_fn)
            elif not args.use_cli:
                model = MSRAMIL(**model_dict, instance_loss_fn=instance_loss_fn)

        else:
            raise NotImplementedError



    model.relocate()
    model.to(device)
    print('Done!')
    print_network(model)

    print('\nInit optimizer ...', end=' ')
    optimizer = get_optim(model, args)

    print('Done!')
    print('\nInit Loaders...', end=' ')

    train_loader = get_split_loader(train_split, training=True, testing=args.testing, weighted=False)
    val_loader = get_split_loader(val_split, testing=args.testing)
    test_loader = get_split_loader(test_split, testing=args.testing)
    print('Done!')

    print('\nSetup EarlyStopping...', end=' ')
    if args.early_stopping:
        early_stopping = EarlyStopping(patience=20, stop_epoch=50, verbose=True)

    else:
        early_stopping = None
    print('Done!')
    global clinical_features
    if args.use_cli:
        clinical_json_address = args.clinical_data
        with open(clinical_json_address, encoding='utf-8') as f:
            json_dict = json.load(f)
        peoples = [i for i in json_dict]
        features = np.array([json_dict[i] for i in json_dict], dtype=np.float32)
        min_max_scaler = preprocessing.MinMaxScaler()

        clinical_features = min_max_scaler.fit_transform(features)

        X_train_minmax = {}
        for i in range(len(peoples)):
            x = {peoples[i]: clinical_features[i]}
            X_train_minmax.update(x)
        clinical_features = X_train_minmax

    for epoch in range(args.max_epochs):
        # scheduler.step()
        if args.model_type in ['MSRAMIL', 'MSRAMIL_C'] and not args.no_inst_cluster:
            if args.use_cli:
                train_loop_MSRAMIL_cli(clinical_features, epoch, model, train_loader, optimizer, args.n_classes,
                                    args.bag_weight, writer, loss_fn)
                stop = validate_MSRAMIL_cli(clinical_features, cur, epoch, model, val_loader, args.n_classes,
                                         early_stopping, writer, loss_fn, args.results_dir)
            else:
                train_loop_MSRAMIL(epoch, model, train_loader, optimizer, args.n_classes, args.bag_weight, writer, loss_fn)
                stop = validate_MSRAMIL(cur, epoch, model, val_loader, args.n_classes,
                                     early_stopping, writer, loss_fn, args.results_dir)

        else:
            if args.use_cli:
                train_loop_cli(clinical_features, epoch, model, train_loader, optimizer, args.n_classes, writer,
                               loss_fn)
                stop = validate_cli(clinical_features, cur, epoch, model, val_loader, args.n_classes,
                                    early_stopping, writer, loss_fn, args.results_dir)
            else:
                train_loop(epoch, model, train_loader, cur, optimizer, args.n_classes, writer, loss_fn)
                stop = validate(cur, epoch, model, val_loader, args.n_classes,
                                early_stopping, writer, loss_fn, args.results_dir)

        if stop:
            break

    if args.early_stopping:
        if args.use_cli:
            model.load_state_dict(torch.load(os.path.join(args.results_dir, "s_{}_checkpoint_cli.pt".format(cur))))
        else:
            model.load_state_dict(torch.load(os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur))))
    else:
        if args.use_cli:
            torch.save(model.state_dict(), os.path.join(args.results_dir, "s_{}_checkpoint_cli.pt".format(cur)))
        else:
            torch.save(model.state_dict(), os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur)))

    if args.use_cli:
        _, val_error, val_auc, _, _, _, _, _, _, _, _, _, _ = summary_cli(clinical_features, model, val_loader,
                                                                       args.n_classes)
        print('Val error: {:.4f}, ROC AUC: {:.4f}'.format(val_error, val_auc))

        results_dict, test_error, test_auc, acc_logger, ci, precision, Recall, F1_score, specificity, label_roc, pred_roc, pred_label, df = summary_cli(
            clinical_features, model, test_loader, args.n_classes)
        print('Test error: {:.4f}, ROC AUC: {:.4f}'.format(test_error, test_auc))
        save_path = os.path.join(writer_dir + 'roc.png')
        fpr, tpr = plot_ROC(label_roc, pred_roc, save_path)
        save_path1 = os.path.join(writer_dir + 'Confusion Matrix.png')
        plot_matrix(label_roc, pred_label, save_path1)

        print('precision:', precision)
        print('Recall:', Recall)
        print('F1_score:', F1_score)
        print('specificity:', specificity)

    else:
        _, val_error, val_auc, _, _, _, _, _, _, _, _, _, _ = summary(model, val_loader, args.n_classes)
        print('Val error: {:.4f}, ROC AUC: {:.4f}'.format(val_error, val_auc))

        results_dict, test_error, test_auc, acc_logger, ci, precision, Recall, F1_score, specificity, label_roc, pred_roc, pred_label, df = summary(
            model, test_loader, args.n_classes)
        print('Test error: {:.4f}, ROC AUC: {:.4f}'.format(test_error, test_auc))

        save_path = os.path.join(writer_dir + 'roc.png')
        fpr, tpr = plot_ROC(label_roc, pred_roc, save_path)
        save_path1 = os.path.join(writer_dir + 'Confusion Matrix.png')
        plot_matrix(label_roc, pred_label, save_path1)


        print('precision:', precision)
        print('Recall:', Recall)
        print('F1_score:', F1_score)
        print('specificity:', specificity)

    for i in range(args.n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))

        if writer:
            writer.add_scalar('final/test_class_{}_acc'.format(i), acc, 0)

    if writer:
        writer.add_scalar('final/val_error', val_error, 0)
        writer.add_scalar('final/val_auc', val_auc, 0)
        writer.add_scalar('final/test_error', test_error, 0)
        writer.add_scalar('final/test_auc', test_auc, 0)
        writer.close()

    return results_dict, test_auc, val_auc, 1 - test_error, 1 - val_error, ci, precision, Recall, F1_score, specificity, fpr, tpr, df


def distillation(y, labels, teacher_scores, temp, alpha):
    return nn.KLDivLoss()(F.log_softmax(y / temp, dim=1), F.softmax(teacher_scores / temp, dim=1)) * (
            temp * temp * 2.0 * alpha) + F.cross_entropy(y, labels) * (1. - alpha)

class SoftTargetCrossEntropy_v2(nn.Module):

    def __init__(self,temp_t=0.1,temp_s=1.):
        super(SoftTargetCrossEntropy_v2, self).__init__()
        self.temp_t = temp_t
        self.temp_s = temp_s

    def forward(self, x: torch.Tensor, target: torch.Tensor, mean: bool= True) -> torch.Tensor:
        loss = torch.sum(-F.softmax(target / self.temp_t,dim=-1) * F.log_softmax(x / self.temp_s, dim=-1), dim=-1)
        if mean:
            return loss.mean()
        else:
            return loss

def train_loop_MSRAMIL(epoch, model, loader, optimizer, n_classes, bag_weight, writer=None, loss_fn=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    inst_logger = Accuracy_Logger(n_classes=n_classes)

    train_loss = 0.
    train_error = 0.
    train_inst_loss = 0.
    inst_count = 0

    for batch_idx, (data, label, slide_id) in enumerate(loader):
        data, label = data.to(device), label.to(device)
        logits1, logits2, logits3, logits, Y_prob, Y_hat, _, instance_dict = model(data, label=label, instance_eval=True)


        acc_logger.log(Y_hat, label)
        loss = (loss_fn(logits1, label) + loss_fn(logits2, label) + loss_fn(logits3, label)) / 3 + loss_fn(logits, label)

        loss_value = loss.item()

        instance_loss = instance_dict['instance_loss']

        inst_count += 1
        instance_loss_value = instance_loss.item()
        train_inst_loss += instance_loss_value

        total_loss = loss * bag_weight + (1 - bag_weight) * instance_loss
        inst_preds = instance_dict['inst_preds']
        inst_labels = instance_dict['inst_labels']
        inst_logger.log_batch(inst_preds, inst_labels)

        train_loss += loss_value  # bag_loss
        if (batch_idx + 1) % 20 == 0:
            print('batch {}, loss: {:.4f}, instance_loss: {:.4f}, weighted_loss: {:.4f}, '.format(batch_idx, loss_value,
                                                                                                  instance_loss_value,
                                                                                                  total_loss.item()) +
                  'label: {}, bag_size: {}'.format(label.item(), data.size(0)))

        error = calculate_error(Y_hat, label)  # bag_level
        train_error += error

        # backward pass
        total_loss.backward()  # bag&ins_loss
        # step
        optimizer.step()
        optimizer.zero_grad()

    # calculate loss and error for epoch
    train_loss /= len(loader)  # bag
    train_error /= len(loader)  # bag

    if inst_count > 0:  # 训练的大图数量
        train_inst_loss /= inst_count
        print('\n')
        for i in range(2):
            acc, correct, count = inst_logger.get_summary(i)
            print('class {} clustering acc {}: correct {}/{}'.format(i, acc, correct, count))

    print('Epoch: {}, train_loss: {:.4f}, train_clustering_loss:  {:.4f}, train_error: {:.4f}'.format(epoch, train_loss,
                                                                                                      train_inst_loss,
                                                                                                      train_error))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        if writer and acc is not None:
            writer.add_scalar('train/class_{}_acc'.format(i), acc, epoch)

    if writer:
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/error', train_error, epoch)
        writer.add_scalar('train/clustering_loss', train_inst_loss, epoch)


def train_loop_MSRAMIL_cli(clinical_features, epoch, model, loader, optimizer, n_classes, bag_weight, writer=None, loss_fn=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    inst_logger = Accuracy_Logger(n_classes=n_classes)

    train_loss = 0.
    train_error = 0.
    train_inst_loss = 0.
    inst_count = 0

    print('\n')
    for batch_idx, (data, label, slide_id) in enumerate(loader):
        data, label = data.to(device), label.to(device)
        for i in slide_id:
            slide_id = i
        patient_id = slide_id
        clinical_features_id = [clinical_features[patient_id]]
        clinical_features_id = torch.from_numpy(np.array(clinical_features_id, dtype=np.float32)).to(device)
        logits1, logits2, logits3, logits, Y_prob, Y_hat, _, instance_dict = model(data, clinical_features_id, label=label, instance_eval=True)

        acc_logger.log(Y_hat, label)
        loss = (loss_fn(logits1, label) + loss_fn(logits2, label) + loss_fn(logits3, label)) / 3 + loss_fn(logits,
                                                                                                           label)

        loss_value = loss.item()

        instance_loss = instance_dict['instance_loss']

        inst_count += 1
        instance_loss_value = instance_loss.item()
        train_inst_loss += instance_loss_value

        total_loss = loss * bag_weight + (1 - bag_weight) * instance_loss
        total_loss = total_loss.to(device)

        inst_preds = instance_dict['inst_preds']
        inst_labels = instance_dict['inst_labels']
        inst_logger.log_batch(inst_preds, inst_labels)


        train_loss += loss_value  # bag_loss
        if (batch_idx + 1) % 20 == 0:
            print('batch {}, loss: {:.4f}, instance_loss: {:.4f}, weighted_loss: {:.4f}, '.format(batch_idx, loss_value,
                                                                                                  instance_loss_value,
                                                                                                  total_loss.item()) +
                  'label: {}, bag_size: {}'.format(label.item(), data.size(0)))

        error = calculate_error(Y_hat, label)  # bag_level
        train_error += error


        total_loss.backward()

        optimizer.step()
        optimizer.zero_grad()


    train_loss /= len(loader)  # bag
    train_error /= len(loader)  # bag

    if inst_count > 0:
        train_inst_loss /= inst_count
        print('\n')
        for i in range(2):
            acc, correct, count = inst_logger.get_summary(i)
            print('class {} clustering acc {}: correct {}/{}'.format(i, acc, correct, count))

    print('Epoch: {}, train_loss: {:.4f}, train_clustering_loss:  {:.4f}, train_error: {:.4f}'.format(epoch, train_loss,
                                                                                                      train_inst_loss,
                                                                                                      train_error))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        if writer and acc is not None:
            writer.add_scalar('train/class_{}_acc'.format(i), acc, epoch)

    if writer:
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/error', train_error, epoch)
        writer.add_scalar('train/clustering_loss', train_inst_loss, epoch)


def train_loop(epoch, model, loader, cur, optimizer, n_classes, writer=None, loss_fn=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    train_loss = 0.
    train_error = 0.

    print('\n')
    for batch_idx, (data, label, slide_id) in enumerate(loader):
        data, label = data.to(device), label.to(device)

        logits1, logits2, logits3, logits, Y_prob, Y_hat, _, _ = model(data)


        acc_logger.log(Y_hat, label)
        loss = (loss_fn(logits1, label) + loss_fn(logits2, label) + loss_fn(logits3, label)) / 3 + loss_fn(logits, label)
        loss_value = loss.item()
        
        train_loss += loss_value
        if (batch_idx + 1) % 20 == 0:
            print('batch {}, loss: {:.4f}, label: {}, bag_size: {}'.format(batch_idx, loss_value, label.item(), data.size(0)))
           
        error = calculate_error(Y_hat, label)
        train_error += error
        
        # backward pass
        loss.backward()
        # step
        optimizer.step()
        optimizer.zero_grad()

    train_loss /= len(loader)
    train_error /= len(loader)

    print('Epoch: {}, train_loss: {:.4f}, train_error: {:.4f}'.format(epoch, train_loss, train_error))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        if writer:
            writer.add_scalar('train/class_{}_acc'.format(i), acc, epoch)

    if writer:
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/error', train_error, epoch)


def train_loop_cli(clinical_features, epoch, model, loader, optimizer, n_classes, writer=None, loss_fn=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    train_loss = 0.
    train_error = 0.

    print('\n')
    for batch_idx, (data, label, slide_id) in enumerate(loader):
        data, label = data.to(device), label.to(device)
        for i in slide_id:
            slide_id = i
        patient_id = slide_id
        clinical_features_id = [clinical_features[patient_id]]
        clinical_features_id = torch.from_numpy(np.array(clinical_features_id, dtype=np.float32)).to(device)

        logits, Y_prob, Y_hat, _, _ = model(data, clinical_features_id)

        acc_logger.log(Y_hat, label)
        loss = loss_fn(logits, label)
        loss_value = loss.item()

        train_loss += loss_value
        if (batch_idx + 1) % 20 == 0:
            print('batch {}, loss: {:.4f}, label: {}, bag_size: {}'.format(batch_idx, loss_value, label.item(),
                                                                           data.size(0)))

        error = calculate_error(Y_hat, label)
        train_error += error

        # backward pass
        loss.backward()
        # step
        optimizer.step()
        optimizer.zero_grad()

    # calculate loss and error for epoch
    train_loss /= len(loader)
    train_error /= len(loader)

    print('Epoch: {}, train_loss: {:.4f}, train_error: {:.4f}'.format(epoch, train_loss, train_error))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        if writer:
            writer.add_scalar('train/class_{}_acc'.format(i), acc, epoch)

    if writer:
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/error', train_error, epoch)

   
def validate(cur, epoch, model, loader, n_classes, early_stopping = None, writer = None, loss_fn = None, results_dir=None):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    # loader.dataset.update_mode(True)
    val_loss = 0.
    val_error = 0.
    
    prob = np.zeros((len(loader), n_classes))
    labels = np.zeros(len(loader))

    with torch.no_grad():
        for batch_idx, (data, label, slide_id) in enumerate(loader):
            data, label = data.to(device, non_blocking=True), label.to(device, non_blocking=True)

            logits1, logits2, logits3, logits, Y_prob, Y_hat, _, _ = model(data)

            acc_logger.log(Y_hat, label)
            
            loss = (loss_fn(logits1, label) + loss_fn(logits2, label) + loss_fn(logits3, label)) / 3 + loss_fn(logits, label)

            prob[batch_idx] = Y_prob.cpu().numpy()
            labels[batch_idx] = label.item()
            
            val_loss += loss.item()
            error = calculate_error(Y_hat, label)
            val_error += error
            

    val_error /= len(loader)
    val_loss /= len(loader)

    if n_classes == 2:
        auc = roc_auc_score(labels, prob[:, 1])
    
    else:
        auc = roc_auc_score(labels, prob, multi_class='ovr')
    
    
    if writer:
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/auc', auc, epoch)
        writer.add_scalar('val/error', val_error, epoch)

    print('\nVal Set, val_loss: {:.4f}, val_error: {:.4f}, auc: {:.4f}'.format(val_loss, val_error, auc))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))     

    if early_stopping:
        assert results_dir
        early_stopping(epoch, val_loss, model, ckpt_name = os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))
        
        if early_stopping.early_stop:
            print("Early stopping")
            return True

    return False


def validate_cli(clinical_features, cur, epoch, model, loader, n_classes, early_stopping=None, writer=None, loss_fn=None, results_dir=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    # loader.dataset.update_mode(True)
    val_loss = 0.
    val_error = 0.

    prob = np.zeros((len(loader), n_classes))
    labels = np.zeros(len(loader))

    with torch.no_grad():
        for batch_idx, (data, label, slide_id) in enumerate(loader):
            data, label = data.to(device, non_blocking=True), label.to(device, non_blocking=True)

            for i in slide_id:
                slide_id = i
            patient_id = slide_id
            clinical_features_id = [clinical_features[patient_id]]
            clinical_features_id = torch.from_numpy(np.array(clinical_features_id, dtype=np.float32)).to(device)

            logits, Y_prob, Y_hat, _, _ = model(data, clinical_features_id)

            acc_logger.log(Y_hat, label)

            loss = loss_fn(logits, label)

            prob[batch_idx] = Y_prob.cpu().numpy()
            labels[batch_idx] = label.item()

            val_loss += loss.item()
            error = calculate_error(Y_hat, label)
            val_error += error

    val_error /= len(loader)
    val_loss /= len(loader)

    if n_classes == 2:
        auc = roc_auc_score(labels, prob[:, 1])

    else:
        auc = roc_auc_score(labels, prob, multi_class='ovr')

    if writer:
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/auc', auc, epoch)
        writer.add_scalar('val/error', val_error, epoch)

    print('\nVal Set, val_loss: {:.4f}, val_error: {:.4f}, auc: {:.4f}'.format(val_loss, val_error, auc))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))

    if early_stopping:
        assert results_dir
        early_stopping(epoch, val_loss, model,
                       ckpt_name=os.path.join(results_dir, "s_{}_checkpoint_cli.pt".format(cur)))
        if early_stopping.early_stop:
            print("Early stopping")
            return True


    return False

def validate_MSRAMIL(cur, epoch, model, loader, n_classes, early_stopping = None, writer = None, loss_fn = None, results_dir = None):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    inst_logger = Accuracy_Logger(n_classes=n_classes)
    val_loss = 0.
    val_error = 0.

    val_inst_loss = 0.
    val_inst_acc = 0.
    inst_count=0
    
    prob = np.zeros((len(loader), n_classes))
    labels = np.zeros(len(loader))
    sample_size = model.k_sample
    with torch.no_grad():
        for batch_idx, (data, label, slide_id) in enumerate(loader):
            data, label = data.to(device), label.to(device)
            logits1, logits2, logits3, logits, Y_prob, Y_hat, _, instance_dict = model(data, label=label,
                                                                                       instance_eval=True)

            acc_logger.log(Y_hat, label)
            loss = loss_fn(logits, label)
            val_loss += loss.item()

            instance_loss = instance_dict['instance_loss']
            
            inst_count+=1
            instance_loss_value = instance_loss.item()
            val_inst_loss += instance_loss_value

            inst_preds = instance_dict['inst_preds']
            inst_labels = instance_dict['inst_labels']
            inst_logger.log_batch(inst_preds, inst_labels)
            # 每个类别有多少数量，正确多少个

            prob[batch_idx] = Y_prob.cpu().numpy()
            labels[batch_idx] = label.item()
            
            error = calculate_error(Y_hat, label)
            val_error += error

    val_error /= len(loader)
    val_loss /= len(loader)

    if n_classes == 2:
        auc = roc_auc_score(labels, prob[:, 1])
        aucs = []
    else:
        aucs = []
        binary_labels = label_binarize(labels, classes=[i for i in range(n_classes)])
        for class_idx in range(n_classes):
            if class_idx in labels:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], prob[:, class_idx])
                aucs.append(calc_auc(fpr, tpr))
            else:
                aucs.append(float('nan'))

        auc = np.nanmean(np.array(aucs))

    print('\nVal Set, val_loss: {:.4f}, val_error: {:.4f}, auc: {:.4f}'.format(val_loss, val_error, auc))
    if inst_count > 0:
        val_inst_loss /= inst_count
        for i in range(2):
            acc, correct, count = inst_logger.get_summary(i)
            print('class {} clustering acc {}: correct {}/{}'.format(i, acc, correct, count))
    
    if writer:
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/auc', auc, epoch)
        writer.add_scalar('val/error', val_error, epoch)
        writer.add_scalar('val/inst_loss', val_inst_loss, epoch)


    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        
        if writer and acc is not None:
            writer.add_scalar('val/class_{}_acc'.format(i), acc, epoch)
     

    if early_stopping:
        assert results_dir
        early_stopping(epoch, val_loss, model, ckpt_name = os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))
        
        if early_stopping.early_stop:
            print("Early stopping")
            return True

    return False


def validate_MSRAMIL_cli(clinical_features, cur, epoch, model, loader, n_classes, early_stopping=None, writer=None, loss_fn=None,
                  results_dir=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    inst_logger = Accuracy_Logger(n_classes=n_classes)
    val_loss = 0.
    val_error = 0.

    val_inst_loss = 0.
    val_inst_acc = 0.
    inst_count = 0

    prob = np.zeros((len(loader), n_classes))
    labels = np.zeros(len(loader))
    sample_size = model.k_sample
    with torch.no_grad():
        for batch_idx, (data, label, slide_id) in enumerate(loader):
            data, label = data.to(device), label.to(device)
            for i in slide_id:
                slide_id = i
            patient_id = slide_id
            clinical_features_id = [clinical_features[patient_id]]
            clinical_features_id = torch.from_numpy(np.array(clinical_features_id, dtype=np.float32)).to(device)
            logits1, logits2, logits3, logits, Y_prob, Y_hat, _, instance_dict = model(data, clinical_features_id, label=label,
                                                                                       instance_eval=True)

            acc_logger.log(Y_hat, label)
            loss = (loss_fn(logits1, label) + loss_fn(logits2, label) + loss_fn(logits3, label)) / 3 + loss_fn(logits,
                                                                                                               label)

            val_loss += loss.item()

            instance_loss = instance_dict['instance_loss']

            inst_count += 1
            instance_loss_value = instance_loss.item()
            val_inst_loss += instance_loss_value

            inst_preds = instance_dict['inst_preds']
            inst_labels = instance_dict['inst_labels']
            inst_logger.log_batch(inst_preds, inst_labels)
            # 每个类别有多少数量，正确多少个

            prob[batch_idx] = Y_prob.cpu().numpy()
            labels[batch_idx] = label.item()

            error = calculate_error(Y_hat, label)
            val_error += error

    val_error /= len(loader)
    val_loss /= len(loader)

    if n_classes == 2:
        auc = roc_auc_score(labels, prob[:, 1])
        aucs = []
    else:
        aucs = []
        binary_labels = label_binarize(labels, classes=[i for i in range(n_classes)])
        for class_idx in range(n_classes):
            if class_idx in labels:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], prob[:, class_idx])
                aucs.append(calc_auc(fpr, tpr))
            else:
                aucs.append(float('nan'))

        auc = np.nanmean(np.array(aucs))

    print('\nVal Set, val_loss: {:.4f}, val_error: {:.4f}, auc: {:.4f}'.format(val_loss, val_error, auc))
    if inst_count > 0:
        val_inst_loss /= inst_count
        for i in range(2):
            acc, correct, count = inst_logger.get_summary(i)
            print('class {} clustering acc {}: correct {}/{}'.format(i, acc, correct, count))

    if writer:
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/auc', auc, epoch)
        writer.add_scalar('val/error', val_error, epoch)
        writer.add_scalar('val/inst_loss', val_inst_loss, epoch)

    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))

        if writer and acc is not None:
            writer.add_scalar('val/class_{}_acc'.format(i), acc, epoch)

    if early_stopping:
        assert results_dir
        early_stopping(epoch, val_loss, model,
                           ckpt_name=os.path.join(results_dir, "s_{}_checkpoint_cli.pt".format(cur)))
        if early_stopping.early_stop:
            print("Early stopping")
            return True

    return False

def summary(model, loader, n_classes):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    model.eval()
    test_loss = 0.
    test_error = 0.

    all_probs = np.zeros((len(loader), n_classes))
    all_labels = np.zeros(len(loader))
    all_preds = np.zeros(len(loader))
    a = loader.dataset.slide_data  # case_id slide_id label [0: n3 n3 0]
    slide_ids = loader.dataset.slide_data['slide_id']  # (0, 'n3') (1, 't6')
    case_ids = loader.dataset.slide_data['case_id']
    patient_results = {}

    label_roc = []
    pred_roc = []
    pred_label = []

    for batch_idx, (data, label, slide_id) in enumerate(loader):
        data, label = data.to(device), label.to(device)
        label_roc += label.tolist()
        slide_id = slide_ids.iloc[batch_idx]
        case_id = case_ids.iloc[batch_idx]
        with torch.no_grad():
            logits1, logits2, logits3, logits, Y_prob, Y_hat, _, _ = model(data)
        pred_label.append(Y_hat.squeeze(-1).squeeze(-1).tolist())
        pred_roc.append(Y_prob[0][1].tolist())
        acc_logger.log(Y_hat, label)
        probs = Y_prob.cpu().numpy()
        all_probs[batch_idx] = probs
        all_labels[batch_idx] = label.item()
        all_preds[batch_idx] = Y_hat.item()

        patient_results.update({slide_id: {'case_id': np.array(case_id), 'slide_id': np.array(slide_id), 'prob': probs, 'label': label.item()}})
        error = calculate_error(Y_hat, label)
        test_error += error

    test_error /= len(loader)
    precision = calculate_precision(label_roc, pred_label)
    Recall = calculate_recall(label_roc, pred_label)
    F1_score = calculate_F1(label_roc, pred_label)
    specificity = calculate_specificity(label_roc, pred_label)
    roc_auc, ci = bootstrap_auc(np.array(label_roc), np.array(pred_roc))

    if n_classes == 2:
        auc = roc_auc_score(all_labels, all_probs[:, 1])
        aucs = []
    else:
        aucs = []
        binary_labels = label_binarize(all_labels, classes=[i for i in range(n_classes)])
        for class_idx in range(n_classes):
            if class_idx in all_labels:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])
                aucs.append(calc_auc(fpr, tpr))
            else:
                aucs.append(float('nan'))

        auc = np.nanmean(np.array(aucs))

    results_dict = {'case_id': case_ids, 'slide_id': slide_ids, 'Y': all_labels, 'Y_hat': all_preds}
    for c in range(n_classes):
        results_dict.update({'p_{}'.format(c): all_probs[:, c]})
    df = pd.DataFrame(results_dict)
    return patient_results, test_error, auc, acc_logger, ci, precision, Recall, F1_score, specificity, label_roc, pred_roc, pred_label, df


def summary_cli(clinical_features, model, loader, n_classes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    model.eval()
    test_loss = 0.
    test_error = 0.

    all_probs = np.zeros((len(loader), n_classes))
    all_labels = np.zeros(len(loader))
    all_preds = np.zeros(len(loader))
    slide_ids = loader.dataset.slide_data['slide_id']
    case_ids = loader.dataset.slide_data['case_id']
    patient_results = {}
    label_roc = []
    pred_roc = []
    pred_label = []
    for batch_idx, (data, label, slide_id) in enumerate(loader):
        data, label = data.to(device), label.to(device)
        label_roc += label.tolist()
        for i in slide_id:
            slide_id = i
        patient_id = slide_id
        clinical_features_id = [clinical_features[patient_id]]
        clinical_features_id = torch.from_numpy(np.array(clinical_features_id, dtype=np.float32)).to(device)
        with torch.no_grad():
            logits1, logits2, logits3, logits, Y_prob, Y_hat, _, _ = model(data, clinical_features_id, label=label,
                                                                                       instance_eval=True)
        pred_label.append(Y_hat.squeeze(-1).squeeze(-1).tolist())
        pred_roc.append(Y_prob[0][1].tolist())
        acc_logger.log(Y_hat, label)
        probs = Y_prob.cpu().numpy()
        all_probs[batch_idx] = probs
        all_labels[batch_idx] = label.item()
        all_preds[batch_idx] = Y_hat.item()

        patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'prob': probs, 'label': label.item()}})
        error = calculate_error(Y_hat, label)
        test_error += error

    test_error /= len(loader)
    precision = calculate_precision(label_roc, pred_label)
    Recall = calculate_recall(label_roc, pred_label)
    F1_score = calculate_F1(label_roc, pred_label)
    specificity = calculate_specificity(label_roc, pred_label)
    roc_auc, ci = bootstrap_auc(np.array(label_roc), np.array(pred_roc))

    if n_classes == 2:
        auc = roc_auc_score(all_labels, all_probs[:, 1])
        aucs = []
    else:
        aucs = []
        binary_labels = label_binarize(all_labels, classes=[i for i in range(n_classes)])
        for class_idx in range(n_classes):
            if class_idx in all_labels:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])
                aucs.append(calc_auc(fpr, tpr))
            else:
                aucs.append(float('nan'))

        auc = np.nanmean(np.array(aucs))

    results_dict = {'case_id': case_ids, 'slide_id': slide_ids, 'Y': all_labels, 'Y_hat': all_preds}
    for c in range(n_classes):
        results_dict.update({'p_{}'.format(c): all_probs[:, c]})
    df = pd.DataFrame(results_dict)
    return patient_results, test_error, auc, acc_logger, ci, precision, Recall, F1_score, specificity, label_roc, pred_roc, pred_label, df
