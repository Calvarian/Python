import numpy as np
import torch as t

def model(x, w, b):
    return x @ w.t() + b


def mse(t1,t2):
    diff = t1 - t2
    diff_sqr = diff * diff
    return t.sum(diff_sqr)/diff.numel()

#Input (temp, rainfall, humidity)

data = np.array([[73, 67, 43],
                [91, 88, 64],
                [87, 134, 58],
                [102, 43, 37],
                [69, 96, 70]], dtype='float32')

#output (apples, oranges)
results = np.array([[56, 70],
                   [81, 101],
                   [119, 133],
                   [22, 37],
                   [103, 119]], dtype='float32')


data = t.from_numpy(data)
results = t.from_numpy(results)

#random weights and biases
w = t.randn(2, 3, requires_grad=True)
b = t.randn(2, requires_grad=True)


for i in range(1000):
    preds = model(data, w, b)
    loss = mse(preds, results)
    loss.backward()
    with t.no_grad():
        w -= w.grad * 1e-5
        b -= b.grad * 1e-5
        w.grad.zero_()
        b.grad.zero_()
        if(i%50 == 0):
            print(loss)
        
# preds = model(data)
# loss = mse(preds, results)
# print(loss)
    
