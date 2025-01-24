import numpy as np
import torch

a = torch.tensor([[1, 2], [3, 4]])
print(a)

a = torch.tensor(np.array([[1, 2], [3, 4]]))
print(a)

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print(device)

a.to(device)
print(a)

b = torch.ones((2,3))
print(b)

b = torch.zeros((2,3))
print(b)

b = torch.full((2,3), fill_value=100)
print(b)

b = torch.eye(2)
print(b)

print("乱数")

# 乱数のシードを設定
torch.manual_seed(42)

# 標準正規分布に従う乱数を生成
print(torch.randn(2))
print()

# 与えられたサイズで, 一様分布[0, 1)に従う乱数を生成
print(torch.rand(2))
print()

c = torch.ones((2,3))

print(c)

print(c.shape)
print(c.size())

print(c.shape[0], c.shape[1])

print("変形")
e = torch.arange(8)
print(e)
print(e.shape)

e = e.reshape(2,4)
print(e)
print(e.shape)

print("view")
e = e.view(1,2,4,1)
print(e)
print(e.shape)