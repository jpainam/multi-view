import os
import torch


def save_checkpoint(state, model, checkpoint='checkpoint', filename='checkpoint.pth'):
    filepath = os.path.join(checkpoint, model + '_' + filename)
    torch.save(state, filepath)