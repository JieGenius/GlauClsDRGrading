## 介绍

GlauClsDRGrading 是 [眼科大模型](https://github.com/JieGenius/OculiChatDA) 的Lagent子项目。

项目最初只计划训练青光眼分类+DR分级两个任务

后期扩展为四个任务

- 青光眼分类
- DR分级 （糖尿病视网膜病变）
- AMD分类 （年龄相关性黄斑变性）
- PM分类 （病理性近视分类）

所有数据均来自开源数据集，各任务的数据集的地址如下：

| 任务    | 数据集名                                                                | 数据集地址                                                                                                                     | 数据集描述                                               |
|-------|---------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------|
| 青光眼分类 | [Refuge](https://refuge.grand-challenge.org/)                       | [Refuge](https://refuge.grand-challenge.org/)                                                                             | 训练验证测试各400张                                         |
| DR分级  | [Aptos2019](https://www.kaggle.com/c/aptos2019-blindness-detection) | [Aptos2019](https://www.kaggle.com/c/aptos2019-blindness-detection)                                                       | 训练3662张，验证1928张                                     |
| AMD分类 | [IChallenge-AMD](https://aistudio.baidu.com/datasetdetail/88462)    | [数据集-part1](https://aistudio.baidu.com/datasetdetail/88462), [数据集-part2](https://aistudio.baidu.com/datasetdetail/177184) | 训练集400张，89张AMD，311张Normal，验证集无标签，测试集未公布图像，故只有200张可用 |
| PM分类  | [IChallenge-PM](https://aistudio.baidu.com/datasetdetail/127091)    | [IChallenge-PM](https://aistudio.baidu.com/datasetdetail/127091)                                                          | 训练集400，验证集400，测试集未公布                                |

## 环境安装

```
conda create -n GlauClsDRGrading python=3.10
conda activate GlauClsDRGrading
pip install -r requirements.txt
pip install wandb
pip install openmim && mim install -e .
wandb login 
```

## 数据准备

准备Refuge数据集

准备kaggle数据集

```bash
pip install kaggle
vim ~/.kaggle/kaggle.json # 设置key
kaggle competitions download -c aptos2019-blindness-detection

unzip aptos2019-blindness-detection.zip

head -n 300 train.csv  > val.csv
mkdir val_images/
tail -n 299 val.csv | cut -d "," -f 1 | xargs -i mv train_images/{}.png val_images/

vim train.csv # 删除第1到第300行，因为这部分被划分到验证集了, 移动到第一行（假设从0开始索引），然后命令模式下执行299dd，回车
```

### 处理前的数据结构如下：
PM和AMD都是从官方直接下载的，保持原始目录结构，REFUGE把官方的图像放到同一个目录，splits下放train/val/test的txt文件（包含图像和标签的映射）
```text
.
|-- amd -> /root/dataset/iChallengeAMD
|   |-- Illustration
|   |   |-- Disc_Fovea_Illustration
|   |   `-- Disc_Masks
|   `-- Training400
|       |-- AMD
|       `-- Non-AMD
|-- aptos -> /root/dataset/aptos
|   |-- test_images
|   |-- train_images
|   `-- val_images
|-- pm -> /root/dataset/iChallengePM
|   |-- PALM-Training400
|   |   |-- PALM-Training400
|   |   |-- PALM-Training400-Annotation-D&F
|   |   |   |-- Disc_Fovea_Illustration
|   |   |   `-- Disc_Masks
|   |   `-- PALM-Training400-Annotation-Lession
|   |       |-- Lesion_Illustration
|   |       `-- Lesion_Masks
|   |           `-- Atrophy
|   |-- PALM-Validation-GT
|   |   |-- Disc_Masks
|   |   `-- Lesion_Masks
|   |       |-- Atrophy
|   |       `-- Detachment
|   `-- PALM-Validation400
`-- refuge -> /root/dataset/REFUGE
    |-- images
    `-- splits

```

### 生成多任务所需的数据集

```bash
cd ~/GlauClsDRGrading
export PYTHONPATH="$(which python):$(pwd)"
python tools/dataset_converters/refuge_aptos_amd_pm.py
```

## 训练

```bash
python tools/train.py configs/convnext/convnext-tiny_b32_refuge_aptos_amd_pm.py --amp --auto-scale-lr
```

## 部署

由于使用了多任务输出，mmdeploy暂时不支持，所以直接使用onnxruntime进行部署
python tools/export_onnx.py

# 模型上传到modelscope

```bash
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash
apt install git-lfs
git clone https://www.modelscope.cn/flyer123/GlauClsDRGrading.git
cd GlauClsDRGrading
git lfs track model.onnx
git add model.onnx
git config --global user.email "you@example.com"
git config --global user.name "Your Name"
git commit -m "add model"
git push
```
