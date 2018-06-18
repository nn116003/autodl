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
    img_pathes = glob.glob(os.path.join(data_dir, "*/*"))
    labs = [tmp.split("/")[-2] for tmp in img_pathes]
    skf = StratifiedKFold(n_splits=10)
    for train_idx, test_idx in skf.split(img_pathes, labs):
        continue

    split_lab = np.zeros(len(img_pathes))
    split_lab[test_idx] = 1
    split_data = pd.DataFrame({"path":img_pathes, "split_lab":split_lab})
    split_data.to_csv(os.path.join(split_output, "split_data.csv"), index=False)
    
    for i in train_idx:
        from_path = img_pathes[i]
        output_dir = os.path.join(train_dir, labs[i])
        mkdir(output_dir)
        shutil.copy(img_pathes[i], output_dir)
    for i in test_idx:
        from_path = img_pathes[i]
        output_dir = os.path.join(val_dir, labs[i])
        mkdir(output_dir)
        shutil.copy(img_pathes[i], output_dir)


if __name__ == "__main__":
    folder_split("./data/train", train_dir = "./data/train2", val_dir = "./data/val2")
