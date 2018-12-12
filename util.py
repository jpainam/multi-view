import os
import torch


def save_checkpoint(state, model, resnet=None, checkpoint='checkpoint', filename='checkpoint.pth'):
    if resnet:
        filepath = os.path.join(checkpoint, model + str(resnet) + '_' + filename)
    else:
        filepath = os.path.join(checkpoint, model + '_' + filename)
    torch.save(state, filepath)