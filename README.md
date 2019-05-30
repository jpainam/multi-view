## Pytorch Re-ID implementation

## Prerequisites

- Python 3.6
- GPU Memory >= 6G
- Numpy
- Pytorch 0.3+

## Data preparation
First, run `python prepare.py` to prepare the dataset. Preparation creates views in each identity. The output 
folder should likes
```python
0743--
      |--view_0
      |--view_1
      |--view_3--
                |--1479_c3s3_080744_06.jpg
                |--1479_c3s3_080694_04

```


## Training
To train the network, use the command 

```bash
python train.py 
`--data` '/home/paul/datasets/market1501/multiviews' 
`--epochs` 100 
`--b 4` (batch size) 
`--lr` 0.0001 (Learning rate)
`--momentum` 0.9 
`--lr-decay-freq` 30 
`--lr-decay` 0.1
```

Most of these parameters are set to default, so you can only run `python train.py`

To resume or use the the pretrained weight `resnet` from ImageNet, add 
```
--resume --pretrained
```

## Testing
To test a trained model, use: 

```
python test.py
python evaluate_gpu (for a gpu evaluation)
python evaluate.py (run on CPU)
```

## Current results

| Datasets | Rank 1 | Rank5 | Rank10 | Rank20 | mAP |
| -------- | ------ | ------| ------ | ------ | ----|
|Market-1501 |    - |    -  |    -  |   -    |     |
| DukeMTMC-ReID|    |       |        |        |     |
| CUHK03 |          |       |        |        |     |

### How to  cite
