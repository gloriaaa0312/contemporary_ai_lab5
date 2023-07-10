import torch
import numpy as np
import random


seed = 10205501437 % (1 << 32)
batch_size = 128
epoch = 5
lr = 1e-4
fine_tune = False


def setup_seed():
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)