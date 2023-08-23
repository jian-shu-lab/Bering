import os
import torch 
import datetime
import warnings
import numpy as np
from logging import Logger
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader

from ._trainer_node import TrainerNode
from ._settings import TRAIN_KEYS

warnings.filterwarnings("ignore")

def is_notebook() -> bool:
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter

def _format_results(metric, digits = 2):
    metric = float(metric)
    metric = np.round(metric * 100, digits)
    return metric

def _draw_loss(
    trainer, record_key, ax, 
    c_train = '#66B2FF', c_validation = '#FF8800', 
    marker = 'o', lwd = 2, mrksize = 5
):
    epochs = trainer.epoch_record[record_key]
    ax.plot(
        epochs, 
        trainer.train_loss_record[record_key], 
        color = c_train, marker = marker, linewidth = lwd, 
        markersize = mrksize, label = 'train'
    )
    ax.plot(
        epochs, 
        trainer.validation_loss_record[record_key], 
        color = c_validation, marker = marker, linewidth = lwd, 
        markersize = mrksize, label = 'test'
    )
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Losses')
    ax.set_title('Training Loss')
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles = handles, labels = labels)

    return ax

def _draw_acc(
    trainer, record_key, ax, 
    col_train = '#66B2FF', col_test = '#FF8800', marker = 'o', lwd = 2, mrksize = 5
):
    epoches = trainer.epoch_record[record_key]

    ax.plot(
        epoches, 
        trainer.acc_train_record[record_key], 
        color = col_train, marker = marker, linewidth = lwd, markersize = mrksize, label = 'train'
    )
    ax.plot(
        epoches, 
        trainer.acc_test_record[record_key], 
        color = col_test, marker = marker, linewidth = lwd, markersize = mrksize, label = 'test'
    )

    ax.set_xlabel('Epoches')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Accuracy')
    
    h, l = ax.get_legend_handles_labels()
    ax.legend(handles = h, labels = l)
    return ax

def _draw_auc(
    trainer, record_key, ax, 
    col_train = '#66B2FF', col_test = '#FF8800', marker = 'o', lwd = 2, mrksize = 5
):
    epochs = trainer.epoch_record[record_key]

    ax.plot(
        epochs, 
        trainer.auc_train_record[record_key], 
        color = col_train, marker = marker, linewidth = lwd, markersize = mrksize, label = 'train'
    )
    ax.plot(
        epochs, 
        trainer.auc_test_record[record_key], 
        color = col_test, marker = marker, linewidth = lwd, markersize = mrksize, label = 'test'
    )

    ax.set_xlabel('Epochs')
    ax.set_ylabel('AUC')
    ax.set_title('AUC')
    
    h, l = ax.get_legend_handles_labels()
    ax.legend(handles = h, labels = l)
    return ax

def _draw_prec(
    trainer, record_key, ax, 
    col_train = '#66B2FF', col_test = '#FF8800', marker = 'o', lwd = 2, mrksize = 5
):
    epochs = trainer.epoch_record[record_key]

    ax.plot(
        epochs, 
        trainer.prec_train_record[record_key], 
        color = col_train, marker = marker, linewidth = lwd, markersize = mrksize, label = 'train'
    )
    ax.plot(
        epochs, 
        trainer.prec_test_record[record_key], 
        color = col_test, marker = marker, linewidth = lwd, markersize = mrksize, label = 'test'
    )

    ax.set_xlabel('Epochs')
    ax.set_ylabel('Precision')
    ax.set_title('Precision')
    
    h, l = ax.get_legend_handles_labels()
    ax.legend(handles = h, labels = l)
    return ax

def _evaluate_metrics(trainer, loader):
    num_correct = 0
    num_samples = 0

    num_error_as_background = 0
    num_error_as_segmented_exist = 0
    num_error_as_segmented_others = 0
    num_error_as_wrong_type = 0
    
    total_labels = np.array([], dtype = np.int16)
    e2_source_labels = np.array([], dtype = np.int16)
    e2_target_labels = np.array([], dtype = np.int16)

    trainer.model.eval()
    
    with torch.no_grad():
        for batch_data in loader.dataset:
            batch_data = batch_data.to(trainer.device)
            batch_label = torch.where(batch_data.y == 1)[1]
            scores = trainer.model(batch_data)
            _, predictions = scores.max(dim = 1)
            
            predictions = predictions.cpu(); batch_label = batch_label.cpu()
            num_correct += (predictions == batch_label).sum()
            num_samples += predictions.size(0)

            # get major metrics
            # from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score
            # auc_score = roc_auc_score(y, preds)
            # precision = average_precision_score(y, preds)
            # accu = accuracy_score(y, preds_binary)

            # different errors
            label_background = int(batch_data.y.shape[1] - 1)
            label_availsegmented = torch.unique(batch_label[batch_label != label_background])
            label_nonavail_segmented = torch.LongTensor(
                np.setdiff1d(np.setdiff1d(torch.arange(int(batch_data.y.shape[1])), label_background), label_availsegmented)
            )

            num_error_as_background += len(np.intersect1d(
                torch.where(predictions == label_background)[0], torch.where(predictions != batch_label)[0]
            ))
            num_error_as_segmented_exist += len(np.intersect1d(
                torch.arange(len(predictions))[torch.isin(predictions, label_availsegmented)], 
                torch.where(batch_label == label_background)[0]
            ))
            num_error_as_segmented_others += len(np.intersect1d(
                torch.arange(len(predictions))[torch.isin(predictions, label_nonavail_segmented)], 
                torch.where(batch_label == label_background)[0]
            ))

            e2 = np.intersect1d(
                torch.where(predictions != label_background)[0], torch.where(batch_label != label_background)[0]
            )
            e2 = np.intersect1d(e2, torch.where(predictions != batch_label)[0])
            num_error_as_wrong_type += len(e2)

            total_labels = np.concatenate([total_labels, batch_label.numpy()])
            e2_source_labels = np.concatenate([e2_source_labels, batch_label[e2].numpy()])
            e2_target_labels = np.concatenate([e2_target_labels, predictions[e2].numpy()])
    
    acc = _format_results(num_correct / num_samples)
    err_seg_as_bg = _format_results(num_error_as_background / num_samples)
    err_seg_as_wrongtype = _format_results(num_error_as_wrong_type / num_samples)
    err_bg_as_seg_avail = _format_results(num_error_as_segmented_exist / num_samples)
    err_bg_as_seg_nonavail = _format_results(num_error_as_segmented_others / num_samples)
    
    cnts_total = np.bincount(total_labels, minlength = label_background + 1)
    cnts_e2_source = np.bincount(e2_source_labels, minlength = label_background + 1)
    cnts_e2_target = np.bincount(e2_target_labels, minlength = label_background + 1)

    return acc, err_seg_as_bg, err_seg_as_wrongtype, err_bg_as_seg_avail, err_bg_as_seg_nonavail, cnts_total, cnts_e2_source, cnts_e2_target

def _get_auc(trainer, loader, image):
    auc_arr = np.array([])
    prec_arr = np.array([])
    accu_arr = np.array([])
    err_pn_arr = np.array([])
    err_np_arr = np.array([])
    with torch.no_grad():
        # for batch_data in loader.dataset:
        for batch_data in loader:
            auc, prec, accu, err_pn, err_np = trainer.predict(batch_data, image)
            auc_arr = np.append(auc_arr, auc)
            prec_arr = np.append(prec_arr, prec)
            accu_arr = np.append(accu_arr, accu)
            err_pn_arr = np.append(err_pn_arr, err_pn)
            err_np_arr = np.append(err_np_arr, err_np)
    return np.mean(auc_arr), np.mean(prec_arr), np.mean(accu_arr), np.mean(err_pn_arr), np.mean(err_np_arr)

def record_init(
    trainer: TrainerNode, 
    record_key: str
):
    # formatted_time = datetime.datetime.now().strftime('%H-%M-%S')
    # if not os.path.isdir(TRAIN_KEYS.FOLDER_RECORD):
    #     os.mkdir(TRAIN_KEYS.FOLDER_RECORD)
    
    trainer.epoch_record = {}
    trainer.train_loss_record = {}
    trainer.validation_loss_record = {}
    trainer.auc_train_record = {}
    trainer.auc_test_record = {}
    trainer.prec_train_record = {}
    trainer.prec_test_record = {}
    trainer.acc_train_record = {}
    trainer.acc_test_record = {}

    trainer.epoch_record[record_key] = []
    trainer.train_loss_record[record_key] = []
    trainer.validation_loss_record[record_key] = []
    trainer.auc_train_record[record_key] = []
    trainer.auc_test_record[record_key] = []
    trainer.prec_train_record[record_key] = []
    trainer.prec_test_record[record_key] = []
    trainer.acc_train_record[record_key] = []
    trainer.acc_test_record[record_key] = []


def record(
    trainer: TrainerNode, 
    image,
    record_key: str, 
    epoch: int, 
    trainer_phase: str,
    train_loss: float, 
    validation_loss: float,
    train_loader: DataLoader, 
    test_loader: DataLoader, 
    logger: Logger, 
    plot_folder: str,
    ax_size: float = 5.0,
    plotting: bool = False,
    plot_name: str = 'performance.png'
):
    trainer.epoch_record[record_key].append(epoch)
    trainer.train_loss_record[record_key].append(train_loss/len(train_loader.dataset))
    trainer.validation_loss_record[record_key].append(validation_loss/len(test_loader.dataset))

    if trainer_phase == 'node':
        acc_train, err1_train, err2_train, err3_train, err4_train, cnts1_train, cnts2_train, cnts3_train = _evaluate_metrics(trainer, train_loader)
        acc_test, err1_test, err2_test, err3_test, err4_test, cnts1_test, cnts2_test, cnts3_test = _evaluate_metrics(trainer, test_loader)
        trainer.acc_train_record[record_key].append(acc_train)
        trainer.acc_test_record[record_key].append(acc_test)

        if plotting:
            fig, axes = plt.subplots(figsize = (ax_size*2,ax_size), nrows = 1, ncols=2, dpi = 300)
            axes[0] = _draw_loss(trainer, record_key, axes[0])
            axes[1] = _draw_acc(trainer, record_key, axes[1])
    else:
        auc_train, prec_train, acc_train, err_pn_train, err_np_train = _get_auc(trainer, train_loader, image)
        auc_test, prec_test, acc_test, err_pn_test, err_np_test = _get_auc(trainer, test_loader, image)
        trainer.auc_train_record[record_key].append(auc_train)
        trainer.auc_test_record[record_key].append(auc_test)
        trainer.prec_train_record[record_key].append(prec_train)
        trainer.prec_test_record[record_key].append(prec_test)
        trainer.acc_train_record[record_key].append(acc_train)
        trainer.acc_test_record[record_key].append(acc_test)

        if plotting:
            fig, axes = plt.subplots(figsize = (ax_size*4, ax_size), ncols = 4, dpi = 300)
            axes[0] = _draw_loss(trainer, record_key, axes[0])
            axes[1] = _draw_prec(trainer, record_key, axes[1])
            axes[2] = _draw_acc(trainer, record_key, axes[2])
            axes[3] = _draw_auc(trainer, record_key, axes[3])

    # logger
    logger.info(f'Train Loss in {record_key} network in epoch {epoch} is {(train_loss/len(train_loader.dataset)):.3f}')
    logger.info(f'Validation Loss in {record_key} network in epoch {epoch} is {(validation_loss/len(test_loader.dataset)):.3f}')
    logger.info(f'Accuracy (training) in {record_key} network in epoch {epoch} is {acc_train:.2f}')
    logger.info(f'Accuracy (testing) in {record_key} network in epoch {epoch} is {acc_test:.2f}')

    # save
    # output_name = TRAIN_KEYS.FOLDER_RECORD + '/' + plot_name
    
    # adjust gaps between subplots
    if plotting:
        plt.subplots_adjust(wspace = 0.5, hspace = 0.5)
        plt.show()
        # save
        output_name = plot_folder + '/' + plot_name
        fig.savefig(output_name, bbox_inches = 'tight')
        # plt.close(fig)

    if trainer_phase == 'edge':
        return auc_train, auc_test, prec_train, prec_test, acc_train, acc_test