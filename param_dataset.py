import torch
from models.vgg import VGG
from torch.utils.data import DataLoader,Dataset

def load_param(param_path):

    p_data = torch.load(param_path)
    # print(p_data)
    return p_data['pdata'].detach()

class ParamDataset(Dataset):

    def __init__(self, param_path):
        self.data = load_param(param_path)
    
    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return int(self.data.shape[0])

# dataset = ParamDataset('/data/bowen/pytorch-AE/param_data/cnn-cifar10/data.pt')