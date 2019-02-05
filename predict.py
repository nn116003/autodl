# coding:utf-8

import numpy as np
import pandas as pd
import seaborn as sns
import os
import shutil

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.models as models

from utils.datasets import ImageFolderWithPathNoLabel

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def predict(img_dir, arch, num_classes, class_labels_file, batch_size=256,
            gpu=True, ckpt='./model_best.pth.tar', output='./predictions.csv'):

    # load model
    net = torch.nn.DataParallel(models.__dict__[arch](num_classes = num_classes))

    if gpu:
        net.cuda()
        
    if gpu:
        checkpoint = torch.load(ckpt)
    else:
        checkpoint = torch.load(ckpt, map_location='cpu')

    net.load_state_dict(checkpoint['state_dict'])
    net.eval()

    # load label file
    lab_ref = pd.read_csv(class_labels_file)

    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize])
                    
    
    # data
    target_data = ImageFolderWithPathNoLabel(img_dir, preprocess)

    hold_num = len(target_data)
    if hold_num < batch_size:
        raise(RuntimeError("data size is less than batch size."))
    
    loader = torch.utils.data.DataLoader(target_data, batch_size=batch_size,
                                         shuffle=False, pin_memory=True)



    # predict
    test_path = []
    test_pred = torch.zeros(hold_num, num_classes)
    start_idx = 0
    for i, (input, path) in enumerate(loader):
        input_var = torch.autograd.Variable(input, volatile=True)
    
        # compute output
        model_output = net(input_var).data.cpu()
        b_size = model_output.size(0)
        test_pred[start_idx:(start_idx + b_size),:] = model_output
        start_idx += b_size
        test_path.extend(path)

    test_pred = pd.DataFrame(test_pred.numpy())
    raw_result = pd.concat((test_pred, pd.DataFrame({"path":test_path})), axis=1)
    raw_result["pred_lab"] = raw_result.iloc[:,:-1].values.argmax(axis=1)
    
    scores = raw_result.iloc[:,:num_classes].values
    ex_scores = np.exp(scores)
    probs = ex_scores / ex_scores.sum(axis=1, keepdims=True)
    probs = pd.DataFrame(probs)
    probs.columns = lab_ref.name.values
    pred_results = pd.concat([probs, raw_result[["path","pred_lab"]]], axis=1)
    pred_results.head()

    pred_results_labname = pred_results.merge(lab_ref, left_on='pred_lab', right_on='id')
    del pred_results_labname['id']
    
    pred_results_labname.to_csv(output, index=False)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='PyTorch ImageClassification Script')
    parser.add_argument('img_dir', metavar='DIR',
                        help='path to dataset')
    parser.add_argument('-ckpt', '--ckpt', default='./model_best.pth.tar', type=str, metavar='PATH',
                    help='path to model file')
    parser.add_argument('-o', '--output', default='./predictions.csv', type=str, metavar='PATH',
                    help='output file path')
    parser.add_argument('-cl', '--class_labels_file', default='./lab_ref.csv', type=str, metavar='PATH',
                    help='path to class_label_file')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    help='model architecture ' +
                        ' (default: resnet18)')
    parser.add_argument('-c', '--num_classes', default=1000, type=int, metavar='N',
                    help='number of class (default: 1000)')
    parser.add_argument('-b', '--batch_size', default=256, type=int,
                        metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument(
        '-g', '--gpu',
        help="use gpu mode",
        action='store_true', default=False)

    args = parser.parse_args()

    predict(**vars(args))



if __name__ == '__main__':
    main()
