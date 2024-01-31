# 构造refuge和aptos多任务学习的数据集
import os
import os.path as osp
import shutil
from os.path import join as pjoin
import json
import pandas as pd

def main():
    refuge_dir = "data/refuge"
    aptos_dir = "data/aptos"
    save_root = "data/refuge_aptos"
    os.makedirs(pjoin(save_root, "splits"), exist_ok=True)
    os.makedirs(pjoin(save_root, "images"), exist_ok=True)

    for split in ["train", "val", "test"]:
        data = {
            "metainfo": {
                "tasks": ["Glau", "DR"]
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

        with open(pjoin(save_root, "splits", split + ".json"), "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
