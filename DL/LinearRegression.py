import torch

def f1():
    x = torch.tensor(3.0)
    w = torch.tensor(4.0, requires_grad=True)
    b = torch.tensor(5.0, requires_grad=True)
    
    y = x * w  + b
    
    y.backward()
    
    print('dy/dx', x.grad)
    print('dy/dw', w.grad)
    print('dy/db', b.grad)

import numpy as np

def f2():
    x = np.array([[1,2],[3,4]])
    
    y = torch.from_numpy(x)
    z = torch.tensor(x)
    print(f'y: {y.dtype} z: {z.dtype}')