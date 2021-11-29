# MVD:Multi-view Variational Distillation

![Network Architecture](https://github.com/FutabaSakuraXD/Farewell-to-Mutual-Information-Variational-Distiilation-for-Cross-Modal-Person-Re-identification/blob/main/images/framework.jpg)

Mindspore implementation for ***Farewell to Mutual Information: Variational Distillation for Cross-Modal Person Re-identification*** in CVPR 2021(Oral). Please read [our paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Tian_Farewell_to_Mutual_Information_Variational_Distillation_for_Cross-Modal_Person_Re-Identification_CVPR_2021_paper.pdf) for a more detailed description of the training procedure.

Original Pytorch implementation can be seen [here](https://github.com/FutabaSakuraXD/Farewell-to-Mutual-Information-Variational-Distiilation-for-Cross-Modal-Person-Re-identification).

## Results

### SYSU-MM01 (all-search mode)

| Metric  | Value  |
| :-----: | :----: |
| Rank-1  | 60.02% |
| Rank-10 | 94.18% |
| Rank-20 | 98.14% |
|   mAP   | 58.80% |

### SYSU-MM01 (indoor-search mode)

| Metric  | Value  |
| :-----: | :----: |
| Rank-1  | 66.05% |
| Rank-10 | 96.59% |
| Rank-20 | 99.38% |
|   mAP   | 72.98% |

### RegDB

|      Mode       | Rank-1 (mAP)  |
| :-------------: | :-----------: |
| Visible-Thermal | 73.2% (71.6%) |
| Thermal-Visible | 71.8% (70.1%) |

### Visualization Results

See [here](https://github.com/FutabaSakuraXD/Farewell-to-Mutual-Information-Variational-Distiilation-for-Cross-Modal-Person-Re-identification#visualization).

***Note**: The aforementioned results have been tested in Python(>=3.7),  Pytorch(>=1.3.0). Please read [our paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Tian_Farewell_to_Mutual_Information_Variational_Distillation_for_Cross-Modal_Person_Re-Identification_CVPR_2021_paper.pdf) for a more detailed description of the training settings (e.g. lr=0.00035 for adam optimizer, etc).

As for mindspore version, although performance is almost comparative with pytorch version, the kl_div functions may have memory leak errors and requires more time to fix. The final version will be updated soon.

## Requirements

- Python==3.7.5
- Mindspore==1.3.0(See [Installation](https://www.mindspore.cn/install/))
- Cuda==10.1
- GPU: we use single Nvidia RTX TITAN for training. GPU mem cost : 20G~22G.

## Get Started

### 1. Prepare the datasets

- (1) SYSU-MM01 Dataset [1]: The SYSU-MM01 dataset can be downloaded from this [website](http://isee.sysu.edu.cn/project/RGBIRReID.htm).
- run `python pre_process_sysu.py` [link](https://github.com/mangye16/Cross-Modal-Re-ID-baseline/blob/master/pre_process_sysu.py) in to prepare the dataset, the training data will be stored in ".npy" format.
- (2) RegDB Dataset [2]: The RegDB dataset can be downloaded from this [website](http://dm.dongguk.edu/link.html) by submitting a copyright form.
- (Named: "Dongguk Body-based Person Recognition Database (DBPerson-Recog-DB1)" on their website).

### 2. Pretrain Model

We use ImageNet-2012 dataset to pretrain resnet-50 backbone. For copyright reasons, checkpoint files cannot be shown publicly. For those who are interesting in our mindspore model, you can get access to  the checkpoint files for academic use only. Please contact zhangzw12319@163.com for application.

Furthermore, our model can still be trained without pretraining. It may lose about  4%-5% precision in CMC & mAP compared to pretrained ones at same training epochs.

### 3. Training

Train a model by

```bash
python train.py --dataset SYSU --data-path "Path-to-dataset" --optim adam --lr 0.00035 --device-target GPU --gpu 0 --pretrain "Path-to-pretrain-model"
```

- `--dataset`: which dataset "SYSU" or "RegDB".
- `--data-path`: manually define the data path(for `SYSU`, path folder must contain `.npy` files, see [`pre_process_sysu.py`](#anchor1) ).
- `--optim`: choose "adam" or "sgd"(default adam)
- `--lr`: initial learning rate( 0.00035 for adam, 0.1 for sgd)
- `--device-target` choose "GPU" or "Ascend"(TODO: add Huawei Model Arts support)
- `--gpu`: which gpu to run(default: 0)
- `--pretrain`: specify resnet-50 checkpoint file path(default "" for no ckpt file)
- `--resume`: specify checkpoint file path for whole model(default "" for no ckpt file, `--resume` loads weights after `--pretrain`, and thus will overwrite `--pretrain` weights). **Please note that mindspore compulsorily requires checkpoint files have `.cpkt` as file suffix, otherwise may trigger errors during loading.**
- `--ckpt`(optional): You can name(tag) your experiments to organize you logs file, e.g. specifying `--ckpt Exp_1` will put your log files into "logs/Exp_1/XXX".Default value is "toy", which means toy experiments(e.g. debugging logs).

Other useful arguments:

We recommend that these hyper-parameters should be kept by default to achieve SOTA performance as reported in the paper. If you want to fine-tune these hyper-parameters to get a better performance, just feel free to modify them.

- `--epochs`: the total number of training epochs, by default 80 Epochs(may be different from original paper).

- `--warmup-steps`: by default 5
- `--start-decay`: the start epoch of lr decay, by default 15.
- `--end-decay`: the ending epoch of lr decay , by default 27.
- `--loss-func`: for ablation study, by default "id+tri"(for now) which is cross-entropy loss plus triplet loss. You can choose from `["id", "id+tri", "id+tri+kldiv"]`. TODO(2021-10-30): for now "id+tri" can achieve 62% rank-1 and 60% mAP on SYSU-MM01, however kl_div functions may cause memory leak in mindspore(we suspect) and we need more time to fix and achieve comparative performance with pytorch in indoor-search mode.

### 4. Evaluation

For now, we integrate evaluation module into train.py. Every epoch we will do evaluations.

Furthermore, you can directly run test.py, with following format

```bash
python test.py --dataset SYSU --data-path "Path-to-dataset" --device-target GPU --gpu 0 --resume "Path-to-saved-checkpoint-file"
```

## Citation

Please kindly cite the references in your publications if it helps your research:

> @inproceedings{VariationalDistillation,
> title={Farewell to Mutual Information Variational Distiilation for Cross-Modal Person Re-identification},
> author={Xudong Tian and Zhizhong Zhang and Shaohui Lin and Yanyun Qu and Yuan Xie and Lizhuang Ma},
> booktitle={Computer Vision and Pattern Recognition},
> year={2021}
> }

## TODO

### Key fixes

- need to fix memory leak issue when use KL-divergence loss func.

### Other fixes

- add Huawei Model Arts support
