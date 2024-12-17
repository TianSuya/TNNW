import argparse, os, sys
import numpy as np
# import imageio
from scipy import ndimage
from param_dataset import ParamDataset, DataLoader
import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import random_split
from models.encoder import medium
import torchvision
from utils import get_interpolations
from evaluate import evaluate_model
from models.vgg import VGG
from rebuild_model import rebuild_model
from tqdm import tqdm
torch.cuda.empty_cache()

torch.manual_seed(42)
device = torch.device('cuda:2')
device_ids = [0,1,2,5,6]
resume = False

if resume == True:
    autoec = torch.load('autoencoder/600.pt')
else:
    autoec = medium(646638, 0.001, 0.1)

autoec = torch.nn.DataParallel(autoec, device_ids=device_ids)
autoec = autoec.cuda(device=device_ids[0])
optimizer = torch.optim.AdamW(autoec.parameters(), lr=1e-2)
epochs = 10000
loss_function = torch.nn.MSELoss(reduction='sum').to(device)
eval_interval = 100
template_model = VGG('Small',10)
train_layer = ['features.12.weight', 'features.12.bias', 'features.12.running_mean', 'features.12.running_var', 'features.12.num_batches_tracked', 'classifier.weight', 'classifier.bias']
print(train_layer)
del template_model

dataset = ParamDataset('/data/bowen/pytorch-AE/param_data/vgg-cifar10.pt')
train_lenth = int(len(dataset) * 0.7)
test_lenth = len(dataset) - train_lenth
train_dataset, test_dataset = random_split(
    dataset=dataset,
    lengths=[train_lenth, test_lenth],
    generator=torch.Generator().manual_seed(0)
)
train_loader = DataLoader(dataset=train_dataset, batch_size = 64, num_workers=4, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size = 64, num_workers=4, shuffle=True)

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
# print(type(test_transform))
task_test_dataset = torchvision.datasets.CIFAR10(root="./data",train=False,transform=test_transform,download=True)
task_test_loader = DataLoader(dataset=task_test_dataset, batch_size = 32, num_workers=4, shuffle=False)

if __name__ == "__main__":
    
    for epoch in range(epochs):
        autoec.train()
        train_loss = []
        for data in train_loader:
            data = data.cuda(device=device_ids[0])
            optimizer.zero_grad()
            recon_batch = autoec(data)
            loss = loss_function(recon_batch, data)
            loss.backward()
            train_loss.append(loss.item())
            optimizer.step()
        print(f'In epoch {epoch}/{epochs}, mean loss is:{sum(train_loss)/len(train_loss)}')

        if epoch % eval_interval != 0:continue 

        with torch.no_grad():
            autoec.eval()
            for batch_idx, data in enumerate(train_loader):
                data = data.to(device)
                recon_batch = autoec(data, True)
                for index in range(5):
                    template_model = VGG('Small',10)
                    now_param = recon_batch[index]
                    print(now_param)
                    now_model = rebuild_model(now_param, template_model, train_layer)
                    now_model = now_model.to(device)
                    acc, test_loss, _ = evaluate_model(now_model, task_test_loader, device)
                    del template_model
                    print(f'Now Process {index}, Acc:{acc}, Test Loss:{test_loss}')
                break
        torch.save(autoec.module, f'autoencoder/{epoch}.pt')
                





                
