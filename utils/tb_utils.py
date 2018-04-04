from tensorboardX import SummaryWriter
import pandas as pd
import numpy as np

import torch
import torch.nn as nn

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class PredHolder(object):
    def __init__(writer, class_num, hold_num = None, lab_ref_dict = None):
        self.writer = writer
        
        self.class_num = class_num
        self.hold_num = hold_num
        
        if self.hold_num is not None:
            self.start_idx = 0
            self.val = {"pred":torch.zeros(hold_num, class_num),
                        "ans":torch.zeros(hold_num),
                        "loss":AverageMeter()}
        else:
            self.val = {"pred":None, "ans":None, "loss":AverageMeter()}

        self.lab_ref_dict = lab_ref_dict

    def update_pred_ans_loss(pred, ans, loss=None):
        
        if self.hold_num is not None:
            l = pred.size[0]
            self.val["pred"][self.start_idx:(self.start_idx + l),:] = pred
            self.val["ans"][self.start_idx:(self.start_idx + l)] = ans
            self.start_idx += l
        else:
            self.val["pred"] = pred
            self.val["ans"] = ans

        if loss is not None:
            self.val["loss"].update(loss, pred.size(0))
        
    def add_loss(iter_idx, name="data/train_loss"):
        self.writer.add_scaler(name, self.val['loss'].avg, iter_idx)
            

    def add_acc(iter_idx, name="data/train_acc"):
        acc = accuracy(self.val["pred"], self.val["ans"])[0]
        self.writer.add_scaler(name, acc, iter_idx)
            
    def add_p_r(iter_idx,
                p_name="data/train_precision_per_class",
                r_name="data/train_recall_per_class"):
        _, pred_idx = self.val["pred"].max(dim=1)
        conf_mat = pd.cross_tab(pred_idx.numpy(), self.val["ans"].numpy()) # pred * ans
        correct_num = np.diag(conf_mat)
        ps = correct_num.astype(float) / conf_mat.sum(axis=1)
        rs = correct_num.astype(float) / conf_mat.sum(axis=0)

        p_dict = {}
        r_dict = {}
        for lab_idx in ps.index:
            p_dict[self.lab_ref_dict[lab_idx]] = ps[lab_idx]
        for lab_idx in rs.index:
            r_dict[self.lab_ref_dict[lab_idx]] = rs[lab_idx]

        self.writer.add_scalar(p_name, p_dict, iter_idx)
        self.writer.add_scalar(r_name, r_dict, iter_idx)

    def add_conf_mat():
        pass

    def add_imgs():
        pass
    
    def add_all():
        pass

    def reset_val():
        if self.hold_num is not None:
            self.start_idx = 0
            self.val = {"pred":torch.zeros(self.hold_num, self.class_num),
                        "ans":torch.zeros(self.hold_num),
                        "loss":AverageMeter()}
        else:
            self.val = {"pred":None, "ans":None, "loss":AverageMeter()}
        
