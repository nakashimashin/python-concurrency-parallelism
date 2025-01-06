import torch
import numpy as np

print("バージョン確認", torch.__version__)

data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)

print("確認1: ",x_data)
print("型確認: ",x_data.dtype)

np_array = np.array(data)
print("確認2: ",np_array)
x_np = torch.from_numpy(np_array)
print("確認3: ",x_np)

x_ones = torch.ones_like(x_data) # x_dataの特性（プロパティ）を維持
print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float) # x_dataのdatatypeを上書き更新
print(f"Random Tensor: \n {x_rand} \n")

tensor = torch.rand(3,4)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")
