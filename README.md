## Pytorch Re-ID implementation

## Prerequisites

- Python 3.6
- GPU Memory >= 6G
- Numpy
- Pytorch 0.3+

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
```bash
@article{DBLP:journals/corr/Ainam2018,
  author    = {Jean-Paul Ainam and
               Qin Ke and
               Guisong Liu and
               Guangchun Luo},
  title     = {Jointly Learning View-Specific Representation and Similarity Metric  for Robust Person Re-Identification},
  year      = {2018},
}
```