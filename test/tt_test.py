import os

import torch
from models.fc_model import FCTextClassifyModel

def t0():
    vocab_size = 500
    net = FCTextClassifyModel(vocab_size,embedding_dim=12,num_classes=2)
    _x = torch.randint(vocab_size, size=(4,8))
    print(_x.shape)
    _r = net(_x)
    print(_r)
    print(_r.shape)


if __name__ == '__main__':
    print(f'当前根目录路径',os.getcwd())
    t0()