# Attention Retractable Transformer (ART) 的中文说明

Jiale Zhang, Yulun Zhang, Jinjin Gu, Yongbing Zhang, Linghe Kong, Xin Yuan。
论文："Accurate Image Restoration with Attention Retractable Transformer", ICLR 2023（Spotlight）

原文与资源：[paper](https://openreview.net/pdf?id=IloMJ5rqfnt) | [arXiv](https://arxiv.org/abs/2210.01427) | [supplementary material](https://openreview.net/attachment?id=IloMJ5rqfnt&name=supplementary_material) | [视觉结果](https://drive.google.com/drive/folders/1b92XHwxuvBLOAiHAjWe-VFKN01hQUiO_?usp=sharing) | [预训练模型](https://drive.google.com/drive/folders/1Sn1sl577Lwancj5SuQtRJk-GwUTRw7lK?usp=share_link)

---
这是基于 PyTorch 的 ART 模型实现。我们的 ART 在以下任务上达到了**最新最优**的性能：

- 双三次插值图像超分辨率（bicubic SR）
- 高斯彩色图像去噪
- 实景图像去噪
- JPEG 压缩伪影去除

摘要：近年来，基于 Transformer 的图像恢复网络由于其参数无关的全局交互在若干任务上优于卷积网络。为降低计算成本，现有方法通常将自注意力限制在不重叠的窗口内计算，但每组 token 往往来自图像的稠密区域，这是一种“稠密注意力”策略，会限制感受野。为了解决该问题，我们提出了 Attention Retractable Transformer (ART)，在网络中交替使用稠密注意力和稀疏注意力模块。稀疏注意力允许来自稀疏区域的 token 互相交互，从而扩大了感受野；稠密与稀疏注意力的交替应用增强了 Transformer 的表示能力，同时对输入图像提供可伸缩的注意力机制。我们在超分辨率、去噪和 JPEG 伪影去除任务上进行了大量实验，定量和可视化结果均优于现有方法。代码和模型见仓库。

---

## 要求

- Python 3.8
- PyTorch >= 1.8.0
- NVIDIA GPU + CUDA

### 安装

```bash
git clone https://github.com/gladzhang/ART.git
cd ART
pip install -r requirements.txt
python setup.py develop
```

## TODO

* [x] 在图像超分辨率上测试
* [x] 在彩色图像去噪上测试
* [x] 在实景图像去噪上测试
* [x] 在 JPEG 压缩伪影去除上测试
* [x] 训练相关代码
* [ ] 更多任务

## 文件与内容概览

1. 模型（Models）
1. 数据集（Datasets）
1. 训练（Training）
1. 测试（Testing）
1. 结果（Results）
1. 引用（Citation）
1. 致谢（Acknowledgement）

---
## 模型

表格中给出了各任务的模型参数、FLOPs、数据集与在对应测试集上的性能指标（PSNR/SSIM），并在 Model Zoo 中提供了预训练模型下载链接。

- 请将下载的模型放到 `experiments/pretrained_models` 目录下，目录内有具体说明。

## 数据集

训练与测试集下载方式在原 README 中已列出（DIV2K、Flickr2K、BSD500、WED、SIDD、DND 等）。请将数据集放入 `datasets/` 目录下，参照仓库内的目录结构。

## 训练说明

以下给出主要任务的训练示例命令，均使用分布式训练（`torch.distributed.launch`）。请根据实际 GPU 数量与配置修改 `--nproc_per_node` 与 `--master_port`。

### 超分辨率（SR）训练示例

```bash
# train ART for SR task, cropped input=64×64, 4 GPUs, batch size=8 per GPU
python -m torch.distributed.launch --nproc_per_node=4 --master_port=2414 basicsr/train.py -opt options/train/train_ART_SR_x2.yml --launcher pytorch
python -m torch.distributed.launch --nproc_per_node=4 --master_port=2414 basicsr/train.py -opt options/train/train_ART_SR_x3.yml --launcher pytorch
python -m torch.distributed.launch --nproc_per_node=4 --master_port=2414 basicsr/train.py -opt options/train/train_ART_SR_x4.yml --launcher pytorch

# ART-S
python -m torch.distributed.launch --nproc_per_node=4 --master_port=2414 basicsr/train.py -opt options/train/train_ART_S_SR_x2.yml --launcher pytorch
``` 

训练日志会输出到 `experiments/` 文件夹中。

### 彩色图像去噪（Gaussian Color DN）训练示例

```bash
python -m torch.distributed.launch --nproc_per_node=4 --master_port=2414 basicsr/train.py -opt options/train/train_ART_ColorDN_level15.yml --launcher pytorch
python -m torch.distributed.launch --nproc_per_node=4 --master_port=2414 basicsr/train.py -opt options/train/train_ART_ColorDN_level25.yml --launcher pytorch
python -m torch.distributed.launch --nproc_per_node=4 --master_port=2414 basicsr/train.py -opt options/train/train_ART_ColorDN_level50.yml --launcher pytorch
```

### 实景去噪（Real DN）训练示例

进入 `realDenoising` 目录，按 README 中提示设置环境（此处项目提供了用于实景去噪训练/测试的特定环境脚本）。

```bash
cd realDenoising
python setup.py develop --no_cuda_ext
python -m torch.distributed.launch --nproc_per_node=8 --master_port=2414 basicsr/train.py -opt options/train_ART_RealDN.yml --launcher pytorch
```

训练完成后，实验日志在 `realDenoising/experiments`。

### JPEG 伪影去除（CAR）训练示例

```bash
python -m torch.distributed.launch --nproc_per_node=4 --master_port=2414 basicsr/train.py -opt options/train/train_ART_CAR_q10.yml --launcher pytorch
python -m torch.distributed.launch --nproc_per_node=4 --master_port=2414 basicsr/train.py -opt options/train/train_ART_CAR_q30.yml --launcher pytorch
python -m torch.distributed.launch --nproc_per_node=4 --master_port=2414 basicsr/train.py -opt options/train/train_ART_CAR_q40.yml --launcher pytorch
```

## 测试说明

同样在 README 中给出了针对各任务的测试命令示例，这里简要概括：

- SR：使用 `basicsr/test.py` 并指定 `options/test/*.yml`。
- 无 GT 时可将待处理图片放在 `datasets/example`，使用 `options/apply/*_without_groundTruth.yml`。
- 彩色去噪 / 实景去噪 / JPEG CAR：分别使用对应的 `options/test` 配置文件或 `realDenoising` 中的脚本。

测试输出默认保存到 `results/` 或 `realDenoising/results/Real_Denoising` 等目录，请参阅具体脚本输出路径。

## 结果

仓库中包含了论文中的表格结果与可视化示例图（详见 `figs/` 目录或上方提供的 Google Drive 链接）。更多结果请参考论文与补充材料。

## 引用

如若使用本代码或模型，请引用：

```
@inproceedings{zhang2023accurate,
  title={Accurate Image Restoration with Attention Retractable Transformer},
  author={Zhang, Jiale and Zhang, Yulun and Gu, Jinjin and Zhang, Yongbing and Kong, Linghe and Yuan, Xin},
  booktitle={ICLR},
  year={2023}
}
```

## 致谢

本工作采用 Apache 2.0 许可证发布。代码基于 [BasicSR](https://github.com/xinntao/BasicSR) 和 [Restormer](https://github.com/swz30/Restormer)，请遵循它们的许可要求。感谢这些优秀工作提供的参考实现。
