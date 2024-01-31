## 介绍
GlauClsDRGrading 是 [眼科大模型](https://github.com/JieGenius/OculiChatDA) 的Lagent子项目。

其主要内容为训练一个青光眼分类和DR分级模型，使用多任务的方式进行训练。

其中，青光眼分类使用Refuge数据集，DR分级使用Kaggle数据集

## 环境安装

"""
conda create -n GlauClsDRGrading python=3.10
conda activate GlauClsDRGrading
pip install -r requirements.txt
pip install wandb
pip install openmim && mim install -e .
wandb login 
"""

## 数据准备
准备Refuge数据集

准备kaggle数据集

```bash
pip install kaggle
vim ~/.kaggle/kaggle.json # 设置key
kaggle competitions download -c aptos2019-blindness-detection

unzip aptos2019-blindness-detection.zip

head -n 300 train.csv  > val.csv
tail -n 299 val.csv | cut -d "," -f 1 | xargs -i mv train_images/{}.png val_images/

vim train.csv # 删除第1到第300行，因为这部分被划分到验证集了。
```

生成多任务所需的数据集

```bash
cd ~/GlauClsDRGrading
export PYTHONPATH="$(which python):$(pwd)"
python tools/dataset_converters/refuge_aptos.py
```

## 训练
```bash

```

## 部署

由于使用了多任务输出，mmdeploy暂时不支持，所以直接使用onnxruntime进行部署
python tools/export_onnx.py

# 模型上传到modescope
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

