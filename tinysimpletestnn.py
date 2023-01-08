#usr/bin/bash python3

from tinygrad.tensor import Tensor

x = Tensor.eye(3, requires_grad=True)
y = Tensor([[2.0,0,2.0]], requires_grad=True)
z = y.matmul(x).sum()
z.backward()

print(x.grad)
print(y.grad)
