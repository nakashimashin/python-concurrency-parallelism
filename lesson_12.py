import numpy as np
import torch

a = torch.tensor([[1, 2], [3, 4]])
print(a)

a = torch.tensor(np.array([[1, 2], [3, 4]]))
print(a)