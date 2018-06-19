import glob
import shutil
import os
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import numpy as np

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def folder_split(data_dir, split_output="./",
                 train_dir = "./data/train", val_dir = "./data/val", test_dir = None):
    img_pathes = np.array(glob.glob(os.path.join(data_dir, "*/*")))
    labs = np.array([tmp.split("/")[-2] for tmp in img_pathes])
    skf = StratifiedKFold(n_splits=10)
    # train 9: test 1
    for train_idx, test_idx in skf.split(img_pathes, labs):
        continue

    # train 8: val 1: test 1
    if test_dir is not None:
        skf2 = StratifiedKFold(n_splits=9)
        for train_train_idx, val_idx in skf2.split(train_idx, labs[train_idx]):
            continue
        val_idx = train_idx[val_idx]
        train_idx = train_idx[train_train_idx]

        
    
    split_lab = np.zeros(len(img_pathes))
    if test_dir is None:
        split_lab[test_idx] = 1
    else:
        split_lab[val_idx] = 1
        split_lab[test_idx] = 2
    split_data = pd.DataFrame({"path":img_pathes, "lab":labs, "split_lab":split_lab})
    split_data.to_csv(os.path.join(split_output, "./result/split_data.csv"), index=False)


    def cp(img_idx, to_dir):
        for i in img_idx:
            from_path = img_pathes[i]
            output_dir = os.path.join(to_dir, labs[i])
            mkdir(output_dir)
            shutil.copy(img_pathes[i], output_dir)

    cp(train_idx, train_dir)
    if test_dir is None:
        cp(test_idx, val_dir)
    else:
        cp(val_idx, val_dir)
        cp(test_idx, test_dir)
    

if __name__ == "__main__":
    mkdir("result")
    folder_split("./data/train", train_dir = "./data/train2", val_dir = "./data/val2", test_dir = "./data/test")
