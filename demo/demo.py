import torch

print(torch.__version__)
print(torch.cuda.is_available())
import os


class STATS(object):
    def __init__(self):
        print(os.getcwd())


a = STATS()

