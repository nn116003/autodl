from tensorboardX import SummaryWriter
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from .utils import *

import torchvision.utils as vutils
from PIL import Image

# TODO last batch

class PredHolder(object):
    def __init__(self,writer, class_num, hold_num = None, lab_ref_dict = None):
        self.writer = writer
        
        self.class_num = class_num
        self.hold_num = hold_num
        
        if self.hold_num is not None:
            self.start_idx = 0
            self.val = {"pred":torch.zeros(hold_num, class_num),
                        "ans":torch.zeros(hold_num).long(),
                        "loss":AverageMeter(),
                        "path":[]}
        else:
            self.val = {"pred":None, "ans":None, "loss":AverageMeter(), "path":[]}

        self.lab_ref_dict = lab_ref_dict

    def update_pred_ans_loss(self, pred, ans, loss=None, path=None):
        
        if self.hold_num is not None:
            l = pred.size(0)
            self.val["pred"][self.start_idx:(self.start_idx + l),:] = pred
            self.val["ans"][self.start_idx:(self.start_idx + l)] = ans
            self.start_idx += l
        else:
            self.val["pred"] = pred
            self.val["ans"] = ans

        if loss is not None:
            self.val["loss"].update(loss, pred.size(0))

        if path is not None:
            self.val["path"].extend(path)
        
    def add_loss(self, iter_idx, name="data/train_loss"):
        self.writer.add_scalar(name, self.val['loss'].avg, iter_idx)
            

    def add_acc(self, iter_idx, name="data/train_acc"):
        acc = accuracy(self.val["pred"], self.val["ans"])[0]
        self.writer.add_scalar(name, acc, iter_idx)
            
    def add_p_r(self, iter_idx,
                p_name="data/train_precision_per_class",
                r_name="data/train_recall_per_class"):
        
        _, pred_idx = self.val["pred"].max(dim=1)
        
        l = self.class_num
        ans_cat = pd.Categorical(self.val["ans"].numpy(), categories=np.arange(l).astype(int))
        pred_cat = pd.Categorical(pred_idx.numpy(), categories=np.arange(l).astype(int))
        
        conf_mat = pd.crosstab(pred_cat, ans_cat) # pred * ans
        correct_num = np.diag(conf_mat)
        ps = correct_num.astype(float) / (conf_mat.sum(axis=1) + 1e-10)
        rs = correct_num.astype(float) / (conf_mat.sum(axis=0) + 1e-10)

        p_dict = {}
        r_dict = {}
        for lab_idx in ps.index:
            p_dict[self.lab_ref_dict[lab_idx]] = ps[lab_idx]
        for lab_idx in rs.index:
            r_dict[self.lab_ref_dict[lab_idx]] = rs[lab_idx]

        self.writer.add_scalars(p_name, p_dict, iter_idx)
        self.writer.add_scalars(r_name, r_dict, iter_idx)

    def add_conf_mat(self):
        pass

    def add_wrong_imgs(self, iter_idx, max_num=3, size=(250, 250), name='Image'):
        _, pred = self.val["pred"].max(dim=1)
        correct = pred.eq(self.val["ans"]).numpy().astype(bool)
        wrong_pathes = np.array(self.val["path"])[~correct]
        use_pathes = np.random.choice(wrong_pathes, max_num, replace=False)
        
        batch = np.zeros((max_num, 3, size[0], size[1])).astype(int)
        for i, path in enumerate(use_pathes):
            print(path)
            img = Image.open(path).convert('RGB').resize(size)
            img_arr = np.array(img).transpose((2,0,1))
            batch[i] = img_arr
            
        grid = vutils.make_grid(torch.from_numpy(batch))
        
        self.writer.add_image(name, grid.numpy().transpose((1,2,0)).astype(np.uint8), iter_idx)
    
    def add_all(self):
        pass

    def save_data(self, outdir="./"):
        prob = pd.DataFrame(self.val["pred"].numpy())
        ans_path = pd.DataFrame({"ans":self.val["ans"].numpy(), "path":self.val["path"]})
        res = pd.concat((prob, ans_path), axis=1)
        res.to_csv(outdir + "latest_res.csv", index=False)
        
        

    def reset_val(self):
        if self.hold_num is not None:
            self.start_idx = 0
            self.val = {"pred":torch.zeros(self.hold_num, self.class_num),
                        "ans":torch.zeros(self.hold_num).long(),
                        "loss":AverageMeter(),
                        "path":[]}
        else:
            self.val = {"pred":None, "ans":None, "loss":AverageMeter(),"path":[]}
        
