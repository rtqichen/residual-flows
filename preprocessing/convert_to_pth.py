import sys
import re
import numpy as np
import torch

img = torch.tensor(np.load(sys.argv[1]))
img = img.permute(0, 3, 1, 2)
torch.save(img, re.sub('.npy$', '.pth', sys.argv[1]))
