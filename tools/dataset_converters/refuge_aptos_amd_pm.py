# 构造refuge和aptos多任务学习的数据集
import os
import os.path as osp
import shutil
from os.path import join as pjoin
import json
import random
import pandas as pd

def main():
    refuge_dir = "data/refuge"
    aptos_dir = "data/aptos"
    amd_dir = "data/amd"
    pm_dir = "data/pm"

    save_root = "data/refuge_aptos_amd_pm"
    os.makedirs(pjoin(save_root, "splits"), exist_ok=True)
    os.makedirs(pjoin(save_root, "images"), exist_ok=True)

    amd_all_lst = os.listdir(pjoin(amd_dir, "Training400", "AMD")) + \
                    os.listdir(pjoin(amd_dir, "Training400", "Non-AMD"))
    random.shuffle(amd_all_lst)
    amd_train_lst = amd_all_lst[:int(len(amd_all_lst) * 0.8)]
    amd_val_lst = amd_all_lst[int(len(amd_all_lst) * 0.8):]
    for split in ["train", "val", "test"]:
        data = {
            "metainfo": {
                "tasks": ["Glau", "DR", "AMD", "PM"]
            },
            "data_list": [

            ]
        }
        refuge_lst = open(pjoin(refuge_dir, "splits", split + ".txt")).readlines()
        for line in refuge_lst:
            name, label = line.strip().split()
            data["data_list"].append({
                "img_path": osp.join("images", name),
                "gt_label": dict(Glau=int(label))
            })
            shutil.copy(pjoin(refuge_dir, "images", name), pjoin(save_root, "images", name))

        aptos_lst = pd.read_csv(pjoin(aptos_dir, split + ".csv"), index_col=0)
        for index, row in aptos_lst.iterrows():
            if "diagnosis" not in row.index:
                # 测试数据集没有标签
                continue
            name, label = index + ".png", row["diagnosis"]
            data["data_list"].append({
                "img_path": osp.join("images", name),
                "gt_label": dict(DR=int(label))
            })
            shutil.copy(pjoin(aptos_dir, split + "_images", name), pjoin(save_root, "images", name))



        if split == "train":
            image_root = pjoin(pm_dir, "PALM-Training400/PALM-Training400")
            for name in os.listdir(image_root):
                if name.endswith(".jpg"):
                    label = 1 if name[0]  == "P" else 0
                    data["data_list"].append({
                        "img_path": osp.join("images", name),
                        "gt_label": dict(PM=label)
                    })
                    shutil.copy(pjoin(image_root, name), pjoin(save_root, "images", name))
        elif split == "val":
            image_root = pjoin(pm_dir, "PALM-Validation400")
            gt = pd.read_excel(pjoin(pm_dir, "PALM-Validation-GT/PM_Label_and_Fovea_Location.xlsx"), index_col=1)

            for name in os.listdir(image_root):
                if name.endswith(".jpg"):
                    label = int(gt.loc[name, "Label"])
                    data["data_list"].append({
                        "img_path": osp.join("images", name),
                        "gt_label": dict(PM=label)
                    })
                    shutil.copy(pjoin(image_root, name), pjoin(save_root, "images", name))

        if split == "train":
            image_root1 = pjoin(amd_dir, "Training400", "AMD")
            image_root2 = pjoin(amd_dir, "Training400", "Non-AMD")

            for name in amd_train_lst:
                if name.endswith(".jpg"):
                    label = 1 if name in os.listdir(image_root1) else 0
                    data["data_list"].append({
                        "img_path": osp.join("images", name),
                        "gt_label": dict(AMD=label)
                    })
                    if name in os.listdir(image_root1):
                        shutil.copy(pjoin(image_root1, name), pjoin(save_root, "images", name))
                    else:
                        shutil.copy(pjoin(image_root2, name), pjoin(save_root, "images", name))
        elif split == "val":
            image_root1 = pjoin(amd_dir, "Training400", "AMD")
            image_root2 = pjoin(amd_dir, "Training400", "Non-AMD")
            for name in amd_val_lst:
                if name.endswith(".jpg"):
                    label = 1 if name in os.listdir(image_root1) else 0
                    data["data_list"].append({
                        "img_path": osp.join("images", name),
                        "gt_label": dict(AMD=label)
                    })
                    if name in os.listdir(image_root1):
                        shutil.copy(pjoin(image_root1, name), pjoin(save_root, "images", name))
                    else:
                        shutil.copy(pjoin(image_root2, name), pjoin(save_root, "images", name))

        with open(pjoin(save_root, "splits", split + ".json"), "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
