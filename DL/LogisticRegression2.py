import torch
import torchvision
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

dataset = MNIST(root='data/', download=True)

#len(dataset)
image, label = dataset[0]
#plt.imshow(image, cmap='gray')
#print(f'Label: {label}')

dataset = MNIST(root='data/',
                train=True,
                transform=transforms.ToTensor())

img_tensor, label = dataset[0]
##print(img_tensor.shape, label)


####
import numpy as np

def split_indices(n, val_pct):
    n_val = int(val_pct*n)
    idxs = np.random.permutation(n)
    return idxs[n_val:], idxs[:n_val]

train_idxs, val_idxs = split_indices(len(dataset), val_pct=0.2)




from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.dataloader import DataLoader

batch_size=100

#training sampler & dataloader
train_sampler = SubsetRandomSampler(train_idxs)
train_loader = DataLoader(dataset,
                          batch_size,
                          sampler=train_sampler)

#validation sampler & dataloader
val_sampler = SubsetRandomSampler(val_idxs)
val_loader = DataLoader(dataset,
                        batch_size,
                        sampler=val_sampler)

import torch.nn as nn

input_size = 28 * 28
num_classes = 10


model = nn.Linear(input_size, num_classes)
##print(model.weight.shape)
model.weight

##print(model.bias.shape)
model.bias



class MnistModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(input_size, num_classes)
        
    def forward(self, xb):
        xb = xb.reshape(-1, 784)
        out = self.linear(xb)
        return out
    
model = MnistModel()

for images, labels in train_loader:
    outputs = model(images)
    break

##print(f'output.shape: {outputs.shape}')
##print(f'Sample outputs: \n {outputs[:2].data}')

import torch.nn.functional as F

probs = F.softmax(outputs, dim=1)
##print(f"Sample probabilities: {probs[:2].data}")
##print('Sum: ',torch.sum(probs[0]).item())

max_probs, preds = torch.max(probs, dim=1)
##print(preds)
##print(max_probs)
##print(labels)

# def accuracy(l1, l2):
#     return torch.sum(l1 == l2).item() / len(l1)

##print(accuracy(preds, labels))

loss_fn = F.cross_entropy
loss = loss_fn(outputs, labels)
print(loss)

learning_rate = 0.001
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)



def loss_batch(model, loss_func, xb, yb, opt=None, metric=None):
    
    # Calculate loss
    preds = model(xb)
    loss = loss_fn(preds, yb)
    
    if opt is not None:
        # Compute gradient
        loss.backward()
        
        # Update params
        opt.step()
        
        # Reset gradient
        opt.zero_grad()
        
    metric_result = None
    
    if metric is not None:
        metric_result = metric(preds, yb)
        
    return loss.item(), len(xb), metric_result


def evaluate(model, loss_fn, valid_dl, metric=None):
    with torch.no_grad():
        
        # Pass each batch through the model
        results = [loss_batch(model, loss_fn, xb, yb, metric=metric)for xb,yb in valid_dl]
        
        # Separate losses, counts and metrics
        losses, nums, metrics = zip(*results)
        
        #Total size of the dataset
        total = np.sum(nums)
        
        # Avg. loss across batches
        avg_loss = np.sum(np.multiply(losses, nums))/total
        avg_metric = None
        if metric is not None:
            avg_metric = np.sum(np.multiply(metrics, nums))/total
        
        return avg_loss, total, avg_metric
    
    
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.sum(preds == labels).item() / len(preds)

print('#### New ####')
val_loss, total, val_acc = evaluate(model, loss_fn, val_loader, metric=accuracy)
print(f'Loss: {val_loss:.4f}, accuracy: {val_acc:.4f}.')

        
def fit(epochs, model, loss_fn, opt, train_dl, valid_dl, metric=None):
    
    for epoch in range(epochs):
        
        # Training
        for xb,yb in train_dl:
            loss,_,_ = loss_batch(model, loss_fn, xb, yb, opt)
            
        # Evaluation
        result = evaluate(model, loss_fn, valid_dl, metric)
        val_loss, total, val_metric = result
        
        # Print progress
        if metric is None:
            print(f'Epoch [{epoch+1}] / [{epochs}], loss: {val_loss:.4f}')
        else:
            print(f'Epoch [{epoch+1}] / [{epochs}], loss: {val_loss:.4f}, {metric.__name__}: {val_metric:.4f}.')
            
            
model = MnistModel()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
fit(20, model, F.cross_entropy, optimizer, train_loader, val_loader, accuracy)
