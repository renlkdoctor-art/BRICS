import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from models.model_MSRAMIL import MSRAMIL, MSRAMIL_C
import os
import pandas as pd
from utils.utils import *
from utils.core_utils import Accuracy_Logger
from sklearn.metrics import roc_auc_score, roc_curve, auc, confusion_matrix
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from sklearn import preprocessing
import sklearn.metrics as metrics

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
    plt.matshow(C, cmap=plt.cm.Blues)
    for i in range(len(C)):
        for j in range(len(C)):
            plt.annotate(C[j, i], xy=(i, j), horizontalalignment='center', verticalalignment='center')
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
    plt.savefig(savepath)


def bootstrap_auc(y_true, y_pred, n_bootstraps=2000, rng_seed=42):
    n_bootstraps = n_bootstraps
    rng_seed = rng_seed
    bootstrapped_scores = []

    rng = np.random.RandomState(rng_seed)
    for i in range(n_bootstraps):
        indices = rng.randint(len(y_pred), size=len(y_pred))
        if len(np.unique(y_true[indices])) > 1:
            score = roc_auc_score(y_true[indices], y_pred[indices])
            bootstrapped_scores.append(score)
    bootstrapped_scores = np.array(bootstrapped_scores)

    print("AUROC: {:0.3f}".format(roc_auc_score(y_true, y_pred)))
    print("Confidence interval for the AUROC score: [{:0.3f} - {:0.3}]".format(
        np.percentile(bootstrapped_scores, (2.5, 97.5))[0], np.percentile(bootstrapped_scores, (2.5, 97.5))[1]))

    return roc_auc_score(y_true, y_pred), np.percentile(bootstrapped_scores, (2.5, 97.5))

def initiate_model(args, ckpt_path):
    print('Init Model')
    model_dict = {"dropout": args.drop_out, 'n_classes': args.n_classes}

    if args.model_size is not None and args.model_type in ['MSRAMIL', 'MSRAMIL_C']:
        model_dict.update({"size_arg": args.model_size, "num_feats":args.num_feats})

    if args.model_type =='MSRAMIL':
        if args.use_cli:
            model = MSRAMIL_C(**model_dict)
        else:
            model = MSRAMIL(**model_dict)

    print_network(model)

    ckpt = torch.load(ckpt_path)

    model_state_dict = ckpt['model']

    print("Model Parameters:")
    for key in model_state_dict.keys():
        print(key)

    model.load_state_dict(model_state_dict)

    model.relocate()
    model.eval()
    return model
def eval(dataset, args, ckpt_path, cur):
    model = initiate_model(args, ckpt_path)
    
    print('Init Loaders')
    loader = get_simple_loader(dataset)
    global clinical_features

    if args.use_cli:
        patient_results, test_error, auc, df, _, ci, precision, Recall, F1_score, specificity, label_roc, pred_roc, pred_label = summary_cli(clinical_features, model, loader, args)
    else:
        patient_results, test_error, auc, df, _, ci, precision, Recall, F1_score, specificity, label_roc, pred_roc, pred_label = summary(model, loader, args)
    writer_dir = os.path.join(args.save_dir, str(cur))
    save_path = os.path.join(writer_dir + 'roc.png')
    plot_ROC(label_roc, pred_roc, save_path)
    save_path1 = os.path.join(writer_dir + 'Confusion Matrix.png')
    plot_matrix(label_roc, pred_label, save_path1)
    print('precision:', precision)
    print('Recall:', Recall)
    print('F1_score:', F1_score)
    print('test_error: ', test_error)
    print('auc: ', auc)
    print('specificity :', specificity )

    return model, patient_results, test_error, auc, df, ci, precision, Recall, F1_score, specificity

def summary(model, loader, args):
    acc_logger = Accuracy_Logger(n_classes=args.n_classes)
    model.eval()
    test_loss = 0.
    test_error = 0.

    all_probs = np.zeros((len(loader), args.n_classes))
    all_labels = np.zeros(len(loader))
    all_preds = np.zeros(len(loader))
    a = loader.dataset.slide_data
    slide_ids = loader.dataset.slide_data['slide_id']
    case_ids = loader.dataset.slide_data['case_id']
    patient_results = {}
    label_roc = []
    pred_roc = []
    pred_label = []
    for batch_idx, (data, label, slide_id) in enumerate(loader):
        data, label = data.to(device), label.to(device)
        slide_id = slide_ids.iloc[batch_idx]
        label_roc += label.tolist()
        with torch.no_grad():
            logits1, logits2, logits3, logits, Y_prob, Y_hat, _, instance_dict = model(data, label=label, instance_eval=True)
        
        acc_logger.log(Y_hat, label)

        acc_logger.log(Y_hat, label)

        probs = Y_prob.cpu().numpy()
        pred_label.append(Y_hat.squeeze(-1).squeeze(-1).tolist())
        pred_roc.append(Y_prob[0][1].tolist())

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

    aucs = []
    if len(np.unique(all_labels)) == 1:
        auc_score = -1

    else: 
        if args.n_classes == 2:
            auc_score = roc_auc_score(all_labels, all_probs[:, 1])
        else:
            binary_labels = label_binarize(all_labels, classes=[i for i in range(args.n_classes)])
            for class_idx in range(args.n_classes):
                if class_idx in all_labels:
                    fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])
                    aucs.append(auc(fpr, tpr))
                else:
                    aucs.append(float('nan'))
            if args.micro_average:
                binary_labels = label_binarize(all_labels, classes=[i for i in range(args.n_classes)])
                fpr, tpr, _ = roc_curve(binary_labels.ravel(), all_probs.ravel())
                auc_score = auc(fpr, tpr)
            else:
                auc_score = np.nanmean(np.array(aucs))

    results_dict = {'case_id': np.array(case_ids), 'slide_id': slide_ids, 'Y': all_labels, 'Y_hat': all_preds}
    for c in range(args.n_classes):
        results_dict.update({'p_{}'.format(c): all_probs[:,c]})
    df = pd.DataFrame(results_dict)
    return patient_results, test_error, auc_score, df, acc_logger, ci, precision, Recall, F1_score, specificity , label_roc, pred_roc, pred_label


def summary_cli(clinical_features, model, loader, args):
    acc_logger = Accuracy_Logger(n_classes=args.n_classes)
    model.eval()
    test_loss = 0.
    test_error = 0.

    all_probs = np.zeros((len(loader), args.n_classes))
    all_labels = np.zeros(len(loader))
    all_preds = np.zeros(len(loader))
    a = loader.dataset.slide_data
    slide_ids = loader.dataset.slide_data['slide_id']
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
            logits1, logits2, logits3, logits, Y_prob, Y_hat, _, instance_dict = model(data, clinical_features_id, label=label,
                                                                                       instance_eval=True)

        acc_logger.log(Y_hat, label)

        probs = Y_prob.cpu().numpy()
        pred_label.append(Y_hat.squeeze(-1).squeeze(-1).tolist())
        pred_roc.append(Y_prob[0][1].tolist())

        all_probs[batch_idx] = probs
        all_labels[batch_idx] = label.item()
        all_preds[batch_idx] = Y_hat.item()

        patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'prob': probs, 'label': label.item()}})

        error = calculate_error(Y_hat, label)
        test_error += error

    del data
    test_error /= len(loader)
    precision = calculate_precision(label_roc, pred_label)
    Recall = calculate_recall(label_roc, pred_label)
    F1_score = calculate_F1(label_roc, pred_label)
    specificity = calculate_specificity(label_roc, pred_label)
    roc_auc, ci = bootstrap_auc(np.array(label_roc), np.array(pred_roc))

    aucs = []
    if len(np.unique(all_labels)) == 1:
        auc_score = -1

    else:
        if args.n_classes == 2:
            auc_score = roc_auc_score(all_labels, all_probs[:, 1])
        else:
            binary_labels = label_binarize(all_labels, classes=[i for i in range(args.n_classes)])
            for class_idx in range(args.n_classes):
                if class_idx in all_labels:
                    fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])
                    aucs.append(auc(fpr, tpr))
                else:
                    aucs.append(float('nan'))
            if args.micro_average:
                binary_labels = label_binarize(all_labels, classes=[i for i in range(args.n_classes)])
                fpr, tpr, _ = roc_curve(binary_labels.ravel(), all_probs.ravel())
                auc_score = auc(fpr, tpr)
            else:
                auc_score = np.nanmean(np.array(aucs))

    results_dict = {'slide_id': slide_ids, 'Y': all_labels, 'Y_hat': all_preds}
    for c in range(args.n_classes):
        results_dict.update({'p_{}'.format(c): all_probs[:, c]})
    df = pd.DataFrame(results_dict)
    return patient_results, test_error, auc_score, df, acc_logger, ci, precision, Recall, F1_score, specificity, label_roc, pred_roc, pred_label

