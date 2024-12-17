import torchvision, torch
from evaluate import evaluate_model
from models.vgg import VGG
from flatten_model import flatten_model
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
from param_dataset import ParamDataset
from rebuild_model import rebuild_model

device = torch.device('cuda:2')

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
param_dataset = ParamDataset('/data/bowen/pytorch-AE/param_data/vgg-cifar10.pt')
test_dataset = torchvision.datasets.CIFAR10(root="./data",train=False,transform=test_transform,download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size = 32, num_workers=4, shuffle=False)
model = VGG('Small', 10).to(device)
train_layer = [name for name, _ in model.named_parameters()]
print(train_layer)

for index in range(10):
    now_param = param_dataset[index].to(device)
    now_model = rebuild_model(now_param, model, train_layer)
    print(evaluate_model(now_model, test_loader, device))
