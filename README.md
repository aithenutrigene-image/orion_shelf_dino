:new: *Please check out our more recent [DINOv2](https://github.com/facebookresearch/dinov2) effort in the same line of work.*

# Self-Supervised Vision Transformers with DINO

PyTorch implementation and pretrained models for DINO. For details, see **Emerging Properties in Self-Supervised Vision Transformers**.  
[[`blogpost`](https://ai.facebook.com/blog/dino-paws-computer-vision-with-self-supervised-transformers-and-10x-more-efficient-training)] [[`arXiv`](https://arxiv.org/abs/2104.14294)] [[`Yannic Kilcher's video`](https://www.youtube.com/watch?v=h3ij3F3cPIk)]

<div align="center">
  <img width="100%" alt="DINO illustration" src=".github/dino.gif">
</div>

# fork 목적과 진행 과정

오리온 매대인식 프로젝트의 경우 라벨링 데이터 확보가 어렵지만 미라벨 데이터의 경우 매월 약 40GB의 데이터가 확보됨  
이를 활용하기 위한 self-supervised 방식을 활용할 경우
* 모여진 대용량 미라벨 이미지를 이용하여 모델 훈련
* 각 과자별 10장에서 ~20장 정도 분류
* KNN classification  

이와 같이 진행시 라벨링에 많은 비용없이 작업 가능  

imagenet 변환 코드 [/dataset/imagenet_format.py](https://github.com/aithenutrigene-image/Orion_Shelf_Classification_Test/blob/main/imagenet_format.py) 를 이용하여 오리온 이미지를 imagenet 형식으로 변경

## base 모델 성능 확인
몇몇 코드 수정 후 아래 코드 진행

첫 진행시
```bash
python eval_knn.py --dump_features "/workspace/dump_features" --data_path "/workspace/imagenet"
```
저장된 파라미터 활용 시
```bash
python eval_knn.py --load_features "/workspace/dump_features" --data_path "/workspace/imagenet"
```

base 결과
| classifier result | Top1 | Top5 |
| ---|---|---|
| 10-NN  | 0.035135662697638105 | 0.11321491313683389 |
| 20-NN  | 0.03903962521959789  | 0.14835057583447198 |
| 100-NN | 0.035135662697638105 | 0.16006246340035135 |
| 200-NN | 0.027327737653718525 | 0.1444466133125122  |

추가 훈련 없이 배포 모델을 바로 사용시 낮은 성능 확인  
이를 해결 하기 위해선 DINO 모델의 훈련 과정이 필요

* 테스트용 pog 이미지 40 GB 활용
* object detection : DINO 진행하여 과자 이미지 추출
* 추출 이미지를 이용하여 self-supervised : DINO 모델 훈련 
* 훈련 모델을 이용 evaluation 진행

## crop image 모델 훈련
207 서버에 pytorch/pytorch:1.12.1-cuda11.3-cudnn8-devel 환경을 이용하여 훈련 진행
```bash
docker run --name orion-dino -it --gpus '"device=1,2"' --ipc=host -v `pwd`/orion_shelf_dino:/workspace pytorch/pytorch:1.12.1-cuda11.3-cudnn8-devel
```

```bash
python -m torch.distributed.launch --nproc_per_node=2 main_dino.py --arch vit_small --data_path /workspace/orion --output_dir /workspace/output --batch_size_per_gpu 128
```


```bash
python eval_knn.py --pretrained_weights "/workspace/output/checkpoint.pth" --dump_features "/workspace/dump_features" --data_path "/workspace/imagenet"
```

| crop image train | classifier result | Top1 | Top5 |
|--- | ---|---|---|
| epoch 0  | 10-NN  |0.027327737653718525 | 0.1795822760101503  |
| epoch 0 | 20-NN  |0.035135662697638105 | 0.22642982627366778 |
| epoch 0 | 100-NN |0.035135662697638105 | 0.30060511419090374 |
| epoch 0 | 200-NN |0.031231700175678313 | 0.35526058949834083 |
| epoch 34 | 10-NN  | 0.031231700175678313 | 0.05075151278547726 |
| epoch 34 | 20-NN  | 0.031231700175678313 | 0.07417528791723599 |
| epoch 34 | 100-NN | 0.031231700175678313 | 0.14835057583447198 |
| epoch 34 | 200-NN | 0.031231700175678313 | 0.14054265079055242 |  


cropping 방식으로 모델 훈련 시 성능 향상이 보이지 않음  
원본 이미지에 바로 적용하는 방식 적용

## original image 모델 훈련

| original image train | classifier result | Top1 | Top5 |
| --- | --- | --- | --- |
| epoch 100 | 10-NN  | 0.023423775131758737 | 0.3630685145422604 |
| epoch 100 | 20-NN  | 0.023423775131758737 | 0.40991606480577786 |
| epoch 100 | 100-NN | 0.10150302557095452  | 0.47628342767909426 |
| epoch 100 | 200-NN | 0.1093109506148741   | 0.5894983408159281 |
| without 알수없음 | 10-NN  | 0.02765486725663717 | 0.38716814159292035 |
| without 알수없음 | 20-NN  | 0.0316055625790139  | 0.43852718078381797 |
| without 알수없음 | 100-NN | 0.1027180783817952  | 0.497787610619469 |
| without 알수없음 | 200-NN | 0.1145701643489254  | 0.604456384323641 |

23.09.23\~24 (토,일) 주말 간 원본 이미지 100epoch 훈련 진행 결과 cropping 보다 좋은 성능을 보이나 결국 10% 대의 낮은 성능을 보임.   
데이터 셋의 한계 (supervised 14\~20%)일지 아니면 현 모델의 한계일지 구분이 안되는 상황.
1369 카테고리 데이터 셋이 아닌 322 데이터 셋을 활용하여 성능 확인 시도

## 322 dataset 모델 성능 확인
```bash
python eval_knn.py --pretrained_weights "/workspace/output/ori/checkpoint.pth" --data_path "/workspace/dataset/imagenet_322"
```

| original image train | classifier result | Top1 | Top5 |
| --- | --- | --- | --- |
| without 알수없음 | 10-NN  | 36.18081094921093 | 56.16591406359108 |
| without 알수없음 | 20-NN  | 35.608211908197944| 57.87440063311764 |
| without 알수없음 | 100-NN | 30.347749173688374| 56.29626181276477 |
| without 알수없음 | 200-NN | 27.29854289837531 | 53.30291885852614 |

```bash
python eval_knn.py --pretrained_weights "/workspace/output/crop/checkpoint.pth" --data_path "/workspace/dataset/imagenet_322"
```

| crop image train | classifier result | Top1 | Top5 |
| --- | --- | --- | --- |
| without 알수없음 | 10-NN  | 93.0636376332573  | 97.64908523811741 |
| without 알수없음 | 20-NN  | 92.93794516083982 | 98.0866812532005 |
| without 알수없음 | 100-NN | 92.21637726362832 | 98.36134258181649 |
| without 알수없음 | 200-NN | 91.7461943112518  | 98.44979284018434 |

crop 데이터셋을 이용할 경우 93% 대의 성능 확인  
base 모델 성능도 비슷하게 나올지 의문

## 322 dataset base 모델 성능 확인

```bash
python eval_knn.py --data_path "/workspace/dataset/imagenet_322"
```

| base | classifier result | Top1 | Top5 |
| --- | --- | --- | --- |
| without 알수없음 | 10-NN  | 85.61985010008846 | 93.53847586239002 |
| without 알수없음 | 20-NN  | 84.96811135422001 | 94.12969601042782 |
| without 알수없음 | 100-NN | 82.5799543782878  | 93.9341743866673 |
| without 알수없음 | 200-NN | 81.00181555793492 | 93.48726781807179 |


## K-NN category_num 변경


실수 발견 322 category 변경면서 knn label 개수를 같이 축소하지 않음.  
허나 성능은 93% 나온 상황
args 에 category_num 을 추가하여 훈련 시작

```bash
python eval_knn.py --data_path "/workspace/dataset/imagenet_322" --category_num 322
```

| base | classifier result | Top1 | Top5 |
| --- | --- | --- | --- |
| without 알수없음 | 10-NN  | 85.61985010008846 | 93.54778641590242 |
| without 알수없음 | 20-NN  | 84.96811135422001 | 94.12969601042782 |
| without 알수없음 | 100-NN | 82.5799543782878  | 93.9341743866673 |
| without 알수없음 | 200-NN | 81.00181555793492 | 93.48726781807179 |

base 기준 성능 변화 없음 (소수점 3번째 부터 차이 발생).  
예측 개수보다 부족할 경우 문제가 있고 많은 경우는 상관 없음으로 보임 

## crop image 300 epoch train

추석 연휴 6일간 TITAN RTX GPU 2개로 훈련 진행 211 epoch 로 진행 train loss 1.92 로 추세를 보았을때 더 학습이 가능하나 1시간당 1.3 epoch 의 훈련 시간을 생각했을 때, 실제 시작되지 않은 프로젝트 임으로 훈련 중단 후 성능 평가

| crop image | classifier result | Top1 | Top5 |
| --- | --- | --- | --- |
| epoch 211 without 알수없음 | 10-NN  | 93.48261254131558 | 97.86788324565896 |
| epoch 211 without 알수없음 | 20-NN  | 93.42209394348494 | 98.26823704669242 |
| epoch 211 without 알수없음 | 100-NN | 93.0543270797449  | 98.57548531260183 |
| epoch 211 without 알수없음 | 200-NN | 92.53293608305013 | 98.71979889204414 |
 
전에 비해 0.4~0.8 사이 향상
성능 분석이 더 필요하나 현상황을 보았을때
모델 변경 시 분류 모델 82% -> 93% 향상 가능

## Pretrained models
You can choose to download only the weights of the pretrained backbone used for downstream tasks, or the full checkpoint which contains backbone and projection head weights for both student and teacher networks. We also provide the backbone in `onnx` format, as well as detailed arguments and training/evaluation logs. Note that `DeiT-S` and `ViT-S` names refer exactly to the same architecture.

<table>
  <tr>
    <th>arch</th>
    <th>params</th>
    <th>k-nn</th>
    <th>linear</th>
    <th colspan="6">download</th>
  </tr>
  <tr>
    <td>ViT-S/16</td>
    <td>21M</td>
    <td>74.5%</td>
    <td>77.0%</td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth">backbone only</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain_full_checkpoint.pth">full ckpt</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deits16.onnx">onnx</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/args.txt">args</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain_log.txt">logs</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain_eval_linear_log.txt">eval logs</a></td>
  </tr>
  <tr>
    <td>ViT-S/8</td>
    <td>21M</td>
    <td>78.3%</td>
    <td>79.7%</td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_deitsmall8_pretrain/dino_deitsmall8_pretrain.pth">backbone only</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_deitsmall8_pretrain/dino_deitsmall8_pretrain_full_checkpoint.pth">full ckpt</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_deitsmall8_pretrain/dino_deits8.onnx">onnx</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_deitsmall8_pretrain/args.txt">args</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_deitsmall8_pretrain/dino_deitsmall8_pretrain_log.txt">logs</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_deitsmall8_pretrain/dino_deitsmall8_pretrain_eval_linear_log.txt">eval logs</a></td>
  </tr>
  <tr>
    <td>ViT-B/16</td>
    <td>85M</td>
    <td>76.1%</td>
    <td>78.2%</td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth">backbone only</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain_full_checkpoint.pth">full ckpt</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitb16.onnx">onnx</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/args.txt">args</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain_log.txt">logs</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain_eval_linear_log.txt">eval logs</a></td>
  </tr>
  <tr>
    <td>ViT-B/8</td>
    <td>85M</td>
    <td>77.4%</td>
    <td>80.1%</td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth">backbone only</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_vitbase8_pretrain/dino_vitbase8_pretrain_full_checkpoint.pth">full ckpt</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_vitbase8_pretrain/dino_vitb8.onnx">onnx</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_vitbase8_pretrain/args.txt">args</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_vitbase8_pretrain/dino_vitbase8_pretrain_log.txt">logs</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_vitbase8_pretrain/dino_vitbase8_pretrain_eval_linear_log.txt">eval logs</a></td>
  </tr>
  <tr>
    <td>ResNet-50</td>
    <td>23M</td>
    <td>67.5%</td>
    <td>75.3%</td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_resnet50_pretrain/dino_resnet50_pretrain.pth">backbone only</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_resnet50_pretrain/dino_resnet50_pretrain_full_checkpoint.pth">full ckpt</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_resnet50_pretrain/dino_resnet50.onnx">onnx</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_resnet50_pretrain/args.txt">args</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_resnet50_pretrain/dino_resnet50_pretrain_log.txt">logs</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_resnet50_pretrain/dino_resnet50_pretrain_eval_linear_log.txt">eval logs</a></td>
  </tr>
</table>

We also release XCiT models ([[`arXiv`](https://arxiv.org/abs/2106.09681)] [[`code`](https://github.com/facebookresearch/xcit)]) trained with DINO:
<table>
  <tr>
    <th>arch</th>
    <th>params</th>
    <th>k-nn</th>
    <th>linear</th>
    <th colspan="5">download</th>
  </tr>
  <tr>
    <td>xcit_small_12_p16</td>
    <td>26M</td>
    <td>76.0%</td>
    <td>77.8%</td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_xcit_small_12_p16_pretrain/dino_xcit_small_12_p16_pretrain.pth">backbone only</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_xcit_small_12_p16_pretrain/dino_xcit_small_12_p16_pretrain_full_checkpoint.pth">full ckpt</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_xcit_small_12_p16_pretrain/args.txt">args</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_xcit_small_12_p16_pretrain/dino_xcit_small_12_p16_pretrain_log.txt">logs</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_xcit_small_12_p16_pretrain/dino_xcit_small_12_p16_pretrain_eval_linear_log.txt">eval</a></td>
  </tr>
  <tr>
    <td>xcit_small_12_p8</td>
    <td>26M</td>
    <td>77.1%</td>
    <td>79.2%</td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_xcit_small_12_p8_pretrain/dino_xcit_small_12_p8_pretrain.pth">backbone only</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_xcit_small_12_p8_pretrain/dino_xcit_small_12_p8_pretrain_full_checkpoint.pth">full ckpt</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_xcit_small_12_p8_pretrain/args.txt">args</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_xcit_small_12_p8_pretrain/dino_xcit_small_12_p8_pretrain_log.txt">logs</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_xcit_small_12_p8_pretrain/dino_xcit_small_12_p8_pretrain_eval_linear_log.txt">eval</a></td>
  </tr>
  <tr>
    <td>xcit_medium_24_p16</td>
    <td>84M</td>
    <td>76.4%</td>
    <td>78.8%</td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_xcit_medium_24_p16_pretrain/dino_xcit_medium_24_p16_pretrain.pth">backbone only</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_xcit_medium_24_p16_pretrain/dino_xcit_medium_24_p16_pretrain_full_checkpoint.pth">full ckpt</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_xcit_medium_24_p16_pretrain/args.txt">args</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_xcit_medium_24_p16_pretrain/dino_xcit_medium_24_p16_pretrain_log.txt">logs</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_xcit_medium_24_p16_pretrain/dino_xcit_medium_24_p16_pretrain_eval_linear_log.txt">eval</a></td>
  </tr>
  <tr>
    <td>xcit_medium_24_p8</td>
    <td>84M</td>
    <td>77.9%</td>
    <td>80.3%</td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_xcit_medium_24_p8_pretrain/dino_xcit_medium_24_p8_pretrain.pth">backbone only</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_xcit_medium_24_p8_pretrain/dino_xcit_medium_24_p8_pretrain_full_checkpoint.pth">full ckpt</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_xcit_medium_24_p8_pretrain/args.txt">args</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_xcit_medium_24_p8_pretrain/dino_xcit_medium_24_p8_pretrain_log.txt">logs</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_xcit_medium_24_p8_pretrain/dino_xcit_medium_24_p8_pretrain_eval_linear_log.txt">eval</a></td>
  </tr>
</table>

### Pretrained models on PyTorch Hub
```python
import torch
vits16 = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
vits8 = torch.hub.load('facebookresearch/dino:main', 'dino_vits8')
vitb16 = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')
vitb8 = torch.hub.load('facebookresearch/dino:main', 'dino_vitb8')
xcit_small_12_p16 = torch.hub.load('facebookresearch/dino:main', 'dino_xcit_small_12_p16')
xcit_small_12_p8 = torch.hub.load('facebookresearch/dino:main', 'dino_xcit_small_12_p8')
xcit_medium_24_p16 = torch.hub.load('facebookresearch/dino:main', 'dino_xcit_medium_24_p16')
xcit_medium_24_p8 = torch.hub.load('facebookresearch/dino:main', 'dino_xcit_medium_24_p8')
resnet50 = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50')
```

## Training

### Documentation
Please install [PyTorch](https://pytorch.org/) and download the [ImageNet](https://imagenet.stanford.edu/) dataset. This codebase has been developed with python version 3.6, PyTorch version 1.7.1, CUDA 11.0 and torchvision 0.8.2. The exact arguments to reproduce the models presented in our paper can be found in the `args` column of the [pretrained models section](https://github.com/facebookresearch/dino#pretrained-models). For a glimpse at the full documentation of DINO training please run:
```
python main_dino.py --help
```

### Vanilla DINO training :sauropod:
Run DINO with ViT-small network on a single node with 8 GPUs for 100 epochs with the following command. Training time is 1.75 day and the resulting checkpoint should reach 69.3% on k-NN eval and 74.0% on linear eval. We provide [training](https://dl.fbaipublicfiles.com/dino/example_runs_logs/dino_vanilla_deitsmall16_log.txt) and [linear evaluation](https://dl.fbaipublicfiles.com/dino/example_runs_logs/dino_vanilla_deitsmall16_eval.txt) logs (with batch size 256 at evaluation time) for this run to help reproducibility.
```
python -m torch.distributed.launch --nproc_per_node=8 main_dino.py --arch vit_small --data_path /path/to/imagenet/train --output_dir /path/to/saving_dir
```

### Multi-node training
We use Slurm and [submitit](https://github.com/facebookincubator/submitit) (`pip install submitit`). To train on 2 nodes with 8 GPUs each (total 16 GPUs):
```
python run_with_submitit.py --nodes 2 --ngpus 8 --arch vit_small --data_path /path/to/imagenet/train --output_dir /path/to/saving_dir
```

<details>
<summary>
DINO with ViT-base network.
</summary>

```
python run_with_submitit.py --nodes 2 --ngpus 8 --use_volta32 --arch vit_base  --data_path /path/to/imagenet/train --output_dir /path/to/saving_dir
```

</details>

### Boosting DINO performance :t-rex:
You can improve the performance of the vanilla run by:
- training for more epochs: `--epochs 300`,
- increasing the teacher temperature: `--teacher_temp 0.07 --warmup_teacher_temp_epochs 30`.
- removing last layer normalization (only safe with `--arch vit_small`): `--norm_last_layer false`,

<details>
<summary>
Full command.
</summary>

```
python run_with_submitit.py --arch vit_small --epochs 300 --teacher_temp 0.07 --warmup_teacher_temp_epochs 30 --norm_last_layer false --data_path /path/to/imagenet/train --output_dir /path/to/saving_dir
```

</details>

The resulting pretrained model should reach 73.3% on k-NN eval and 76.0% on linear eval. Training time is 2.6 days with 16 GPUs. We provide [training](https://dl.fbaipublicfiles.com/dino/example_runs_logs/dino_boost_deitsmall16_log.txt) and [linear evaluation](https://dl.fbaipublicfiles.com/dino/example_runs_logs/dino_boost_deitsmall16_eval.txt) logs (with batch size 256 at evaluation time) for this run to help reproducibility.

### ResNet-50 and other convnets trainings
This code also works for training DINO on convolutional networks, like ResNet-50 for example. We highly recommend to adapt some optimization arguments in this case. For example following is a command to train DINO on ResNet-50 on a single node with 8 GPUs for 100 epochs. We provide [training logs](https://dl.fbaipublicfiles.com/dino/example_runs_logs/dino_rn50_log.txt) and [final checkpoint](https://dl.fbaipublicfiles.com/dino/example_runs_logs/dino_rn50_checkpoint.pth) for this run.
```
python -m torch.distributed.launch --nproc_per_node=8 main_dino.py --arch resnet50 --optimizer sgd --lr 0.03 --weight_decay 1e-4 --weight_decay_end 1e-4 --global_crops_scale 0.14 1 --local_crops_scale 0.05 0.14 --data_path /path/to/imagenet/train --output_dir /path/to/saving_dir
```

## Self-attention visualization
You can look at the self-attention of the [CLS] token on the different heads of the last layer by running:
```
python visualize_attention.py
```

<div align="center">
  <img width="100%" alt="Self-attention from a Vision Transformer with 8x8 patches trained with DINO" src=".github/attention_maps.png">
</div>

## Self-attention video generation
You can generate videos like the one on the blog post with `video_generation.py`.

https://user-images.githubusercontent.com/46140458/116817761-47885e80-ab68-11eb-9975-d61d5a919e13.mp4

Extract frames from input video and generate attention video:
```
python video_generation.py  --pretrained_weights dino_deitsmall8_pretrain.pth \
    --input_path input/video.mp4 \
    --output_path output/ \
    --fps 25
```

Use folder of frames already extracted and generate attention video:
```
python video_generation.py  --pretrained_weights dino_deitsmall8_pretrain.pth \
    --input_path output/frames/ \
    --output_path output/ \
    --resize 256 \
```

Only generate video from folder of attention maps images:
```
python video_generation.py --input_path output/attention \
    --output_path output/ \
    --video_only \
    --video_format avi
```


## Evaluation: k-NN classification on ImageNet
To evaluate a simple k-NN classifier with a single GPU on a pre-trained model, run:
```
python -m torch.distributed.launch --nproc_per_node=1 eval_knn.py --data_path /path/to/imagenet
```
If you choose not to specify `--pretrained_weights`, then DINO reference weights are used by default. If you want instead to evaluate checkpoints from a run of your own, you can run for example:
```
python -m torch.distributed.launch --nproc_per_node=1 eval_knn.py --pretrained_weights /path/to/checkpoint.pth --checkpoint_key teacher --data_path /path/to/imagenet 
```

## Evaluation: Linear classification on ImageNet
To train a supervised linear classifier on frozen weights on a single node with 8 gpus, run:
```
python -m torch.distributed.launch --nproc_per_node=8 eval_linear.py --data_path /path/to/imagenet
```

We release the logs and weights from evaluating the different models:

<table>
  <tr>
    <th>arch</th>
    <th>top-1 ImageNet</th>
    <th colspan="2">linear evaluation</th>
  </tr>
  <tr>
    <td>ViT-S/16</td>
    <td>77.0%</td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_linearweights.pth">linear weights</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain_eval_linear_log.txt">logs</a></td>
  </tr>
  <tr>
    <td>ViT-S/8</td>
    <td>79.7%</td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_deitsmall8_pretrain/dino_deitsmall8_linearweights.pth">linear weights</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_deitsmall8_pretrain/dino_deitsmall8_pretrain_eval_linear_log.txt">logs</a></td>
  </tr>
  <tr>
    <td>ViT-B/16</td>
    <td>78.2%</td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_linearweights.pth">linear weights</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain_eval_linear_log.txt">logs</a></td>
  </tr>
  <tr>
    <td>ViT-B/8</td>
    <td>80.1%</td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_vitbase8_pretrain/dino_vitbase8_linearweights.pth">linear weights</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_vitbase8_pretrain/dino_vitbase8_pretrain_eval_linear_log.txt">logs</a></td>
  </tr>
  <tr>
    <td>xcit_small_12_p16</td>
    <td>77.8%</td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_xcit_small_12_p16_pretrain/dino_xcit_small_12_p16_linearweights.pth">linear weights</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_xcit_small_12_p16_pretrain/dino_xcit_small_12_p16_pretrain_eval_linear_log.txt">logs</a></td>
  </tr>
  <tr>
    <td>xcit_small_12_p8</td>
    <td>79.2%</td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_xcit_small_12_p8_pretrain/dino_xcit_small_12_p8_linearweights.pth">linear weights</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_xcit_small_12_p8_pretrain/dino_xcit_small_12_p8_pretrain_eval_linear_log.txt">logs</a></td>
  </tr>
  <tr>
    <td>xcit_medium_24_p16</td>
    <td>78.8%</td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_xcit_medium_24_p16_pretrain/dino_xcit_medium_24_p16_linearweights.pth">linear weights</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_xcit_medium_24_p16_pretrain/dino_xcit_medium_24_p16_pretrain_eval_linear_log.txt">logs</a></td>
  </tr>
  <tr>
    <td>xcit_medium_24_p8</td>
    <td>80.3%</td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_xcit_medium_24_p8_pretrain/dino_xcit_medium_24_p8_linearweights.pth">linear weights</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_xcit_medium_24_p8_pretrain/dino_xcit_medium_24_p8_pretrain_eval_linear_log.txt">logs</a></td>
  </tr>
  <tr>
    <td>ResNet-50</td>
    <td>75.3%</td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_resnet50_pretrain/dino_resnet50_linearweights.pth">linear weights</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_resnet50_pretrain/dino_resnet50_pretrain_eval_linear_log.txt">logs</a></td>
  </tr>
</table>

You can check the performance of the pretrained weights on ImageNet validation set by running the following command lines:
```
python eval_linear.py --evaluate --arch vit_small --patch_size 16 --data_path /path/to/imagenet/train
```

```
python eval_linear.py --evaluate --arch vit_small --patch_size 8 --data_path /path/to/imagenet/train
```

```
python eval_linear.py --evaluate --arch vit_base --patch_size 16 --n_last_blocks 1 --avgpool_patchtokens true --data_path /path/to/imagenet/train
```

```
python eval_linear.py --evaluate --arch vit_base --patch_size 8 --n_last_blocks 1 --avgpool_patchtokens true --data_path /path/to/imagenet/train
```

```
python eval_linear.py --evaluate --arch resnet50 --data_path /path/to/imagenet/train
```

## Evaluation: DAVIS 2017 Video object segmentation
Please verify that you're using pytorch version 1.7.1 since we are not able to reproduce the results with most recent pytorch 1.8.1 at the moment.

**Step 1: Prepare DAVIS 2017 data**  
```
cd $HOME
git clone https://github.com/davisvideochallenge/davis-2017 && cd davis-2017
./data/get_davis.sh
```

**Step 2: Video object segmentation**  
```
python eval_video_segmentation.py --data_path $HOME/davis-2017/DAVIS/ --output_dir /path/to/saving_dir
```

**Step 3: Evaluate the obtained segmentation**  
```
git clone https://github.com/davisvideochallenge/davis2017-evaluation $HOME/davis2017-evaluation
python $HOME/davis2017-evaluation/evaluation_method.py --task semi-supervised --results_path /path/to/saving_dir --davis_path $HOME/davis-2017/DAVIS/
```

## Evaluation: Image Retrieval on revisited Oxford and Paris
Step 1: Prepare revisited Oxford and Paris by following [this repo](https://github.com/filipradenovic/revisitop).

Step 2: Image retrieval (if you do not specify weights with `--pretrained_weights` then by default [DINO weights pretrained on Google Landmark v2 dataset](https://dl.fbaipublicfiles.com/dino/dino_vitsmall16_googlelandmark_pretrain/dino_vitsmall16_googlelandmark_pretrain.pth) will be used).

Paris:
```
python -m torch.distributed.launch --use_env --nproc_per_node=1 eval_image_retrieval.py --imsize 512 --multiscale 1 --data_path /path/to/revisited_paris_oxford/ --dataset rparis6k
```

Oxford:
```
python -m torch.distributed.launch --use_env --nproc_per_node=1 eval_image_retrieval.py --imsize 224 --multiscale 0 --data_path /path/to/revisited_paris_oxford/ --dataset roxford5k
```

## Evaluation: Copy detection on Copydays
Step 1: Prepare [Copydays dataset](https://lear.inrialpes.fr/~jegou/data.php#copydays).

Step 2 (opt): Prepare a set of image distractors and a set of images on which to learn the whitening operator.
In our paper, we use 10k random images from YFCC100M as distractors and 20k random images from YFCC100M (different from the distractors) for computing the whitening operation.

Step 3: Run copy detection:
```
python -m torch.distributed.launch --use_env --nproc_per_node=1 eval_copy_detection.py --data_path /path/to/copydays/ --whitening_path /path/to/whitening_data/ --distractors_path /path/to/distractors/
```
We report result on the strong subset. For example in the stdout from the command above we get: `eval on strong mAP=0.858`.

## License
This repository is released under the Apache 2.0 license as found in the [LICENSE](LICENSE) file.

## Citation
If you find this repository useful, please consider giving a star :star: and citation :t-rex::
```
@inproceedings{caron2021emerging,
  title={Emerging Properties in Self-Supervised Vision Transformers},
  author={Caron, Mathilde and Touvron, Hugo and Misra, Ishan and J\'egou, Herv\'e  and Mairal, Julien and Bojanowski, Piotr and Joulin, Armand},
  booktitle={Proceedings of the International Conference on Computer Vision (ICCV)},
  year={2021}
}
```
