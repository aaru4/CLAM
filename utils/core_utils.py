import numpy as np
import torch
import torch.nn as nn
from utils.survival_utils import survival_ce_loss, survival_nll_loss, cox_ph_loss
from utils.utils import *
import os
from utils.utils import collate_MIL_survival
from dataset_modules.dataset_generic import save_splits
from models.model_mil import MIL_fc, MIL_fc_mc
from models.model_clam import CLAM_MB, CLAM_SB
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import auc as calc_auc
from lifelines.utils import concordance_index

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Accuracy_Logger(object):
    """Accuracy logger"""
    def __init__(self, n_classes):
        super().__init__()
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
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=20, stop_epoch=50, verbose=False):
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, epoch, val_loss, model, ckpt_name='checkpoint.pt'):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, ckpt_name):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), ckpt_name)
        self.val_loss_min = val_loss


def train(datasets, cur, args):
    """train for a single fold"""
    print('\nTraining Fold {}!'.format(cur))
    writer_dir = os.path.join(args.results_dir, str(cur))
    if not os.path.isdir(writer_dir):
        os.mkdir(writer_dir)

    if args.log_data:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(writer_dir, flush_secs=15)
    else:
        writer = None

    if args.task in ['task_4_survival_binned_ce', 'task_5_survival_nll']:
        _collate = collate_MIL_survival
    elif args.task == 'task_6_survival_cox':
        _collate = collate_MIL_cox
    else:
        _collate = collate_MIL
    
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
    print('Done!')

    print('\nInit Model...', end=' ')
    model_dict = {"dropout": args.drop_out,
                  'n_classes': args.n_classes,
                  "embed_dim": args.embed_dim}

    if args.model_size is not None and args.model_type != 'mil':
        model_dict.update({"size_arg": args.model_size})

    if args.model_type in ['clam_sb', 'clam_mb']:
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

        if args.model_type == 'clam_sb':
            model = CLAM_SB(**model_dict, instance_loss_fn=instance_loss_fn)
        elif args.model_type == 'clam_mb':
            model = CLAM_MB(**model_dict, instance_loss_fn=instance_loss_fn)
        else:
            raise NotImplementedError
    else:
        if args.n_classes > 2:
            model = MIL_fc_mc(**model_dict)
        else:
            model = MIL_fc(**model_dict)

    _ = model.to(device)
    print('Done!')
    print_network(model)

    print('\nInit optimizer ...', end=' ')
    optimizer = get_optim(model, args)
    print('Done!')

    print('\nInit Loaders...', end=' ')
    train_loader = get_split_loader(train_split, training=True, testing=args.testing, weighted=args.weighted_sample, collate_fn=_collate)
    val_loader = get_split_loader(val_split, testing=args.testing, collate_fn=_collate)
    test_loader = get_split_loader(test_split, testing=args.testing, collate_fn=_collate)
    print('Done!')

    print('\nSetup EarlyStopping...', end=' ')
    if args.early_stopping:
        early_stopping = EarlyStopping(patience=20, stop_epoch=50, verbose=True)
    else:
        early_stopping = None
    print('Done!')

    # ── Survival tasks use dedicated loops ───────────────────────────────────
    if args.task == 'task_4_survival_binned_ce':
        surv_loss_fn = survival_ce_loss
        for epoch in range(args.max_epochs):
            train_loop_survival(epoch, model, train_loader, optimizer,
                                args.n_classes, args.bag_weight, writer, surv_loss_fn)
            stop = validate_survival(cur, epoch, model, val_loader,
                                     args.n_classes, early_stopping, writer,
                                     surv_loss_fn, args.results_dir)
            if stop:
                break

    elif args.task == 'task_5_survival_nll':
        surv_loss_fn = survival_nll_loss
        for epoch in range(args.max_epochs):
            train_loop_survival(epoch, model, train_loader, optimizer,
                                args.n_classes, args.bag_weight, writer, surv_loss_fn)
            stop = validate_survival(cur, epoch, model, val_loader,
                                     args.n_classes, early_stopping, writer,
                                     surv_loss_fn, args.results_dir)
            if stop:
                break

    elif args.task == 'task_6_survival_cox':
        for epoch in range(args.max_epochs):
            train_loop_cox(epoch, model, train_loader, optimizer, args, writer)
            stop = validate_cox(cur, epoch, model, val_loader,
                                early_stopping, writer, args.results_dir)
            if stop:
                break

    # ── Standard classification tasks ────────────────────────────────────────
    else:
        for epoch in range(args.max_epochs):
            if args.model_type in ['clam_sb', 'clam_mb'] and not args.no_inst_cluster:
                train_loop_clam(epoch, model, train_loader, optimizer,
                                args.n_classes, args.bag_weight, writer, loss_fn)
                stop = validate_clam(cur, epoch, model, val_loader, args.n_classes,
                                     early_stopping, writer, loss_fn, args.results_dir)
            else:
                train_loop(epoch, model, train_loader, optimizer,
                           args.n_classes, writer, loss_fn)
                stop = validate(cur, epoch, model, val_loader, args.n_classes,
                                early_stopping, writer, loss_fn, args.results_dir)
            if stop:
                break

    if args.early_stopping:
        model.load_state_dict(torch.load(os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur))))
    else:
        torch.save(model.state_dict(), os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur)))
    
    if args.task in ['task_4_survival_binned_ce', 'task_5_survival_nll', 'task_6_survival_cox']:
        return {}, 0.0, 0.0, 0.0, 0.0
    
    _, val_error, val_auc, _ = summary(model, val_loader, args.n_classes)
    print('Val error: {:.4f}, ROC AUC: {:.4f}'.format(val_error, val_auc))
    
    results_dict, test_error, test_auc, acc_logger = summary(model, test_loader, args.n_classes)
    print('Test error: {:.4f}, ROC AUC: {:.4f}'.format(test_error, test_auc))

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

    return results_dict, test_auc, val_auc, 1-test_error, 1-val_error


def train_loop_survival(epoch, model, loader, optimizer, n_classes, bag_weight, writer, loss_fn):
    model.train()
    train_loss = 0.
    all_risk_scores = []
    all_survival_bins = []
    all_events = []

    print('\n')
    for batch_idx, (data, label) in enumerate(loader):
        data  = data.to(device)
        label = label.to(device)

        survival_bin = label[:, 0].long()
        event        = label[:, 1].float()

        logits, Y_prob, Y_hat, A_raw, instance_dict = model(data, label=survival_bin, instance_eval=True)

        loss = loss_fn(logits, survival_bin, event)

        if instance_dict:
            instance_loss = instance_dict['instance_loss']
            loss = bag_weight * loss + (1 - bag_weight) * instance_loss

        train_loss += loss.item()

        if (batch_idx + 1) % 20 == 0:
            print('batch {}, loss: {:.4f}, bin: {}, bag_size: {}'.format(
                batch_idx, loss.item(), survival_bin.item(), data.size(0)))

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        risk = torch.sigmoid(logits).detach().cpu().numpy()
        all_risk_scores.append(risk.mean(axis=1))
        all_survival_bins.extend(survival_bin.cpu().numpy())
        all_events.extend(event.cpu().numpy())

    train_loss /= len(loader)

    try:
        c_index = concordance_index(
            all_survival_bins,
            [-r for r in np.concatenate(all_risk_scores)],
            all_events
        )
    except Exception:
        c_index = 0.5

    print('Epoch: {}, train_loss: {:.4f}, train_c_index: {:.4f}'.format(epoch, train_loss, c_index))
    if writer:
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/c_index', c_index, epoch)

    return train_loss, c_index


def validate_survival(cur, epoch, model, loader, n_classes, early_stopping, writer, loss_fn, results_dir):
    model.eval()
    val_loss = 0.
    all_risk_scores = []
    all_survival_bins = []
    all_events = []

    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(loader):
            data  = data.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)

            survival_bin = label[:, 0].long()
            event        = label[:, 1].float()

            logits, Y_prob, Y_hat, _, instance_dict = model(data, label=survival_bin, instance_eval=True)
            loss = loss_fn(logits, survival_bin, event)
            val_loss += loss.item()

            risk = torch.sigmoid(logits).cpu().numpy()
            all_risk_scores.append(risk.mean(axis=1))
            all_survival_bins.extend(survival_bin.cpu().numpy())
            all_events.extend(event.cpu().numpy())

    val_loss /= len(loader)

    try:
        c_index = concordance_index(
            all_survival_bins,
            [-r for r in np.concatenate(all_risk_scores)],
            all_events
        )
    except Exception:
        c_index = 0.5

    print('\nVal Set, val_loss: {:.4f}, val_c_index: {:.4f}'.format(val_loss, c_index))
    if writer:
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/c_index', c_index, epoch)

    if early_stopping:
        assert results_dir
        early_stopping(epoch, val_loss, model,
                       ckpt_name=os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))
        if early_stopping.early_stop:
            print("Early stopping")
            return True

    return False


def train_loop_clam(epoch, model, loader, optimizer, n_classes, bag_weight, writer=None, loss_fn=None):
    model.train()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    inst_logger = Accuracy_Logger(n_classes=n_classes)

    train_loss = 0.
    train_error = 0.
    train_inst_loss = 0.
    inst_count = 0

    print('\n')
    for batch_idx, (data, label) in enumerate(loader):
        data, label = data.to(device), label.to(device)
        logits, Y_prob, Y_hat, _, instance_dict = model(data, label=label, instance_eval=True)

        acc_logger.log(Y_hat, label)
        loss = loss_fn(logits, label)
        loss_value = loss.item()

        instance_loss = instance_dict['instance_loss']
        inst_count += 1
        instance_loss_value = instance_loss.item()
        train_inst_loss += instance_loss_value

        total_loss = bag_weight * loss + (1 - bag_weight) * instance_loss

        inst_preds = instance_dict['inst_preds']
        inst_labels = instance_dict['inst_labels']
        inst_logger.log_batch(inst_preds, inst_labels)

        train_loss += loss_value
        if (batch_idx + 1) % 20 == 0:
            print('batch {}, loss: {:.4f}, instance_loss: {:.4f}, weighted_loss: {:.4f}, '.format(
                batch_idx, loss_value, instance_loss_value, total_loss.item()) +
                'label: {}, bag_size: {}'.format(label.item(), data.size(0)))

        error = calculate_error(Y_hat, label)
        train_error += error

        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    train_loss /= len(loader)
    train_error /= len(loader)

    if inst_count > 0:
        train_inst_loss /= inst_count
        print('\n')
        for i in range(2):
            acc, correct, count = inst_logger.get_summary(i)
            print('class {} clustering acc {}: correct {}/{}'.format(i, acc, correct, count))

    print('Epoch: {}, train_loss: {:.4f}, train_clustering_loss: {:.4f}, train_error: {:.4f}'.format(
        epoch, train_loss, train_inst_loss, train_error))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        if writer and acc is not None:
            writer.add_scalar('train/class_{}_acc'.format(i), acc, epoch)

    if writer:
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/error', train_error, epoch)
        writer.add_scalar('train/clustering_loss', train_inst_loss, epoch)


def train_loop(epoch, model, loader, optimizer, n_classes, writer=None, loss_fn=None):
    model.train()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    train_loss = 0.
    train_error = 0.

    print('\n')
    for batch_idx, (data, label) in enumerate(loader):
        data, label = data.to(device), label.to(device)

        logits, Y_prob, Y_hat, _, _ = model(data)

        acc_logger.log(Y_hat, label)
        loss = loss_fn(logits, label)
        loss_value = loss.item()

        train_loss += loss_value
        if (batch_idx + 1) % 20 == 0:
            print('batch {}, loss: {:.4f}, label: {}, bag_size: {}'.format(
                batch_idx, loss_value, label.item(), data.size(0)))

        error = calculate_error(Y_hat, label)
        train_error += error

        loss.backward()
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


def validate(cur, epoch, model, loader, n_classes, early_stopping=None, writer=None, loss_fn=None, results_dir=None):
    model.eval()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    val_loss = 0.
    val_error = 0.

    prob = np.zeros((len(loader), n_classes))
    labels = np.zeros(len(loader))

    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(loader):
            data, label = data.to(device, non_blocking=True), label.to(device, non_blocking=True)

            logits, Y_prob, Y_hat, _, _ = model(data)

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
                       ckpt_name=os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))
        if early_stopping.early_stop:
            print("Early stopping")
            return True

    return False


def validate_clam(cur, epoch, model, loader, n_classes, early_stopping=None, writer=None, loss_fn=None, results_dir=None):
    model.eval()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    inst_logger = Accuracy_Logger(n_classes=n_classes)
    val_loss = 0.
    val_error = 0.
    val_inst_loss = 0.
    inst_count = 0

    prob = np.zeros((len(loader), n_classes))
    labels = np.zeros(len(loader))
    sample_size = model.k_sample

    with torch.inference_mode():
        for batch_idx, (data, label) in enumerate(loader):
            data, label = data.to(device), label.to(device)
            logits, Y_prob, Y_hat, _, instance_dict = model(data, label=label, instance_eval=True)
            acc_logger.log(Y_hat, label)

            loss = loss_fn(logits, label)
            val_loss += loss.item()

            instance_loss = instance_dict['instance_loss']
            inst_count += 1
            instance_loss_value = instance_loss.item()
            val_inst_loss += instance_loss_value

            inst_preds = instance_dict['inst_preds']
            inst_labels = instance_dict['inst_labels']
            inst_logger.log_batch(inst_preds, inst_labels)

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
                       ckpt_name=os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))
        if early_stopping.early_stop:
            print("Early stopping")
            return True

    return False


def summary(model, loader, n_classes):
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    model.eval()
    test_loss = 0.
    test_error = 0.

    all_probs = np.zeros((len(loader), n_classes))
    all_labels = np.zeros(len(loader))

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}

    for batch_idx, (data, label) in enumerate(loader):
        data, label = data.to(device), label.to(device)
        slide_id = slide_ids.iloc[batch_idx]
        with torch.inference_mode():
            logits, Y_prob, Y_hat, _, _ = model(data)

        acc_logger.log(Y_hat, label)
        probs = Y_prob.cpu().numpy()
        all_probs[batch_idx] = probs
        all_labels[batch_idx] = label.item()

        patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'prob': probs, 'label': label.item()}})
        error = calculate_error(Y_hat, label)
        test_error += error

    test_error /= len(loader)

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

    return patient_results, test_error, auc, acc_logger

def train_loop_cox(epoch, model, loader, optimizer, args, writer=None):
    """
    Training loop for Cox proportional hazards survival model.

    This loop:
      - Runs the model forward on each bag/patient
      - Collects predicted risk scores (logits)
      - Buffers multiple samples together
      - Computes Cox partial likelihood on the buffered set
      - Performs a gradient update

    Important:
    Cox loss requires *multiple patients at once* because each event is
    evaluated relative to a "risk set" of other patients. That is why
    we buffer several mini-batches before stepping the optimizer.
    """

    model.train()  # Set model to training mode (enables dropout, etc.)

    train_loss = 0.   # Accumulate loss across updates
    n_updates  = 0    # Count how many optimizer steps we take

    # Buffer to temporarily store risk scores and survival info
    cox_buffer = []

    # Minimum of 2 is required (Cox needs comparisons between patients)
    cox_batch_size = max(2, args.cox_batch_size)

    for batch_idx, (data, time_to_event, event) in enumerate(loader):

        # Move tensors to GPU/CPU device
        data          = data.to(device)
        time_to_event = time_to_event.to(device)
        event         = event.to(device)

        # Forward pass
        # logits = model output = log hazard score (NOT probability)
        logits, _, _, _, _ = model(data)

        # Flatten logits to shape (N,)
        # Each value is a log hazard score for one patient/bag
        risk = logits.view(-1)

        # Store current batch predictions and survival info
        # We cannot compute Cox loss on a single patient —
        # it must compare multiple patients together.
        cox_buffer.append((risk,
                           time_to_event.view(-1),
                           event.view(-1)))

        # Decide whether to perform an optimizer step
        # We step when:
        #   - buffer reaches desired size
        #   - OR we are at the last batch of the epoch
        should_step = (
            len(cox_buffer) >= cox_batch_size
            or (batch_idx == len(loader) - 1)
        )

        if should_step:

            # Concatenate buffered predictions into one larger group
            # This creates a mini-cohort for Cox partial likelihood
            risk_all  = torch.cat([item[0] for item in cox_buffer], dim=0)
            time_all  = torch.cat([item[1] for item in cox_buffer], dim=0)
            event_all = torch.cat([item[2] for item in cox_buffer], dim=0)

            # Compute Cox negative partial log-likelihood
            loss = cox_ph_loss(risk_all, time_all, event_all)

            # Standard optimization step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_value  = loss.item()
            n_updates  += 1

            # Clear buffer after update
            cox_buffer = []

        else:
            # If we don't step yet, continue accumulating patients
            continue

        train_loss += loss_value

        # Print progress every 50 batches
        if (batch_idx + 1) % 50 == 0:
            print(f'batch {batch_idx}, train_loss: {loss_value:.4f}')

    # Average loss across optimizer steps (not batches!)
    train_loss /= max(1, n_updates)

    print(f'Epoch: {epoch}, train_surv_loss: {train_loss:.4f}')

    # Log to TensorBoard if writer is provided
    if writer:
        writer.add_scalar('train/surv_loss', train_loss, epoch)

    return train_loss


def validate_cox(cur, epoch, model, loader, early_stopping, writer, results_dir):
    """Validation loop for Cox PH — computes loss and c-index."""
    model.eval()
    val_loss = 0.
    all_risks = []
    all_times = []
    all_events = []

    with torch.no_grad():
        for batch_idx, (data, time_to_event, event) in enumerate(loader):
            data          = data.to(device, non_blocking=True)
            time_to_event = time_to_event.to(device, non_blocking=True)
            event         = event.to(device, non_blocking=True)

            logits, _, _, _, _ = model(data)
            risk = logits.view(-1)

            # Need at least 2 samples for Cox loss
            if risk.shape[0] > 1:
                loss = cox_ph_loss(risk, time_to_event.view(-1), event.view(-1))
                val_loss += loss.item()

            all_risks.extend(risk.cpu().numpy())
            all_times.extend(time_to_event.cpu().numpy())
            all_events.extend(event.cpu().numpy())

    val_loss /= max(1, len(loader))

    try:
        c_index = concordance_index(all_times, [-r for r in all_risks], all_events)
    except Exception:
        c_index = 0.5

    print(f'\nVal Set, val_loss: {val_loss:.4f}, val_c_index: {c_index:.4f}')
    if writer:
        writer.add_scalar('val/surv_loss', val_loss, epoch)
        writer.add_scalar('val/c_index', c_index, epoch)

    if early_stopping:
        assert results_dir
        early_stopping(epoch, val_loss, model,
                       ckpt_name=os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))
        if early_stopping.early_stop:
            print("Early stopping")
            return True

    return False
