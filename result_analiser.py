# coding:utf-8

import numpy as np
import pandas as pd
import seaborn as sns
import os
import shutil
import argparse

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

from utils.datasets import ImageFolderWithPath
def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)



parser = argparse.ArgumentParser(description='result analizer')
parser.add_argument('--data', '-d', metavar='DIR', default='./data',
                    help='data dir')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    help='model architecture ' +
                        ' (default: resnet18)')
parser.add_argument('-c', '--num_classes', default=1000, type=int, metavar='N',
                    help='number of class (default: 1000)')
parser.add_argument('-b', '--batch_size', default=48, type=int,
                        metavar='N', help='mini-batch size (default: 48)')
args = parser.parse_args()



arch = args.arch
num_classes = args.num_classes
data_dir = args.data
batch_size = args.batch_size
workers = 4

mkdir("./result")

###############################
#### predict for test data ####
###############################

#model 読み込み
model = models.__dict__[arch](num_classes = num_classes).cuda()
model = torch.nn.DataParallel(model).cuda()
checkpoint = torch.load("./model_best.pth.tar")
model.load_state_dict(checkpoint['state_dict'])
model.eval()

# データセット
testdir = os.path.join(data_dir, 'test')
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
test_dataset = ImageFolderWithPath(
    testdir,
    transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ]))
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=batch_size, shuffle=False,
    num_workers=workers, pin_memory=True)

# 予測
hold_num = len(test_dataset)
test_pred = torch.zeros(hold_num, num_classes)
test_ans = torch.zeros(hold_num).long()
test_path = []
start_idx = 0
for i, ((input, target), path) in enumerate(test_loader):
    target = target.cuda(async=True)
    input_var = torch.autograd.Variable(input, volatile=True)
    target_var = torch.autograd.Variable(target, volatile=True)
    
    # compute output
    output = model(input_var).data.cpu()
    b_size = output.size(0)
    test_pred[start_idx:(start_idx + b_size),:] = output
    test_ans[start_idx:(start_idx + b_size)] = target
    start_idx += b_size
    test_path.extend(path)


test_pred = pd.DataFrame(test_pred.numpy())
ans_path = pd.DataFrame({"ans":test_ans.numpy(), "path":test_path})
data = pd.concat((test_pred, ans_path), axis=1)


########################
#### output results ####
########################

# accuracy
top3_pred = (-data.iloc[:,:-2]).values.argsort(axis=1)[:, :3]
data["pred_lab"] = top3_pred[:,0]
is_correct = top3_pred == data.ans.values.reshape(-1,1)
acc = is_correct[:,0].mean()
top3_acc = is_correct.max(axis=1).mean()
f = open('./result/accuracy.txt', 'w')
f.write("top1acc:" + str(acc) + "\n")
f.write("top3acc:" + str(top3_acc) + "\n")
f.close()

# 予測ラベルとクラス名対応表
lab_ref = pd.read_csv("lab_ref.csv")
lab_ref_dict = {}
for i in range(lab_ref.shape[0]):
    lab_ref_dict[i] = lab_ref.name[i]

# 予測値、結果データ
scores = data.iloc[:,:lab_ref.shape[0]].values
ex_scores = np.exp(scores)
probs = ex_scores / ex_scores.sum(axis=1, keepdims=True)
probs = pd.DataFrame(probs)
probs.columns = lab_ref.name.values
pred_results = pd.concat([probs, data[["ans", "path","pred_lab"]]], axis=1)
pred_results.head()

pred_results_labname = pred_results.merge(lab_ref, left_on='pred_lab', right_on='id')
del pred_results_labname['id']

pred_results_labname.to_csv("./result/prediction.csv", index=False)

# 誤判別画像フォルダ
mkdir("./result/fail_imgs")
data["correct"] = data.ans == data.pred_lab
fail_data = data.iloc[~data.correct.values,:]
fail_data.reset_index(inplace=True)
fail_data.head()
for i in range(fail_data.shape[0]):
    ans_lab = lab_ref_dict[fail_data.ans[i]]
    output_dir = os.path.join("./result/fail_imgs", ans_lab)
    mkdir(output_dir)
    pred_lab = lab_ref_dict[fail_data.pred_lab[i]]
    from_path = fail_data.path[i]
    to_name = str(pred_lab) + "_" + os.path.basename(from_path)
    to_path = os.path.join(output_dir, to_name)
    shutil.copy(from_path, to_path)

# confusion matrix (ans * pred)
l = lab_ref.shape[0]

ans_cat = pd.Categorical(data.ans.values, categories=np.arange(l).astype(int))
pred_cat = pd.Categorical(data.iloc[:,:l].values.argmax(axis=1), categories=np.arange(l).astype(int))
conf_mat = pd.crosstab(ans_cat, pred_cat, dropna=False) # ans * pred

nums = conf_mat.values.astype(float)
prec = np.diag(nums)/(nums.sum(axis=0) + 1e-10)
recall = np.diag(nums)/(nums.sum(axis=1) + 1e-10)

conf_result = np.zeros((l+1, l+1))
conf_result[:l, :l] = conf_mat.values
conf_result[l,:l] = prec
conf_result[:l,l] = recall
conf_result = pd.DataFrame(conf_result)
conf_result.columns =  list(lab_ref.name.values) + ["recall"]
conf_result.index = list(lab_ref.name.values) + ["precision"]
conf_result.to_csv("result/conf_mat.csv")
