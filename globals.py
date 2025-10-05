import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import timm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
# from tqdm.notebook import tqdm
from tqdm import tqdm


if __name__ == '__main__':

    print('System Version:', sys.version)
    print('Pytorch version:', torch.__version__)
    print('Torchvision version:', torchvision.__version__)
    print('Numpy version:', np.__version__)
    print('Pandas version:', pd.__version__)




