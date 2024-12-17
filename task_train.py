import torchvision, torch
from evaluate import evaluate_model
from models.vgg import VGG
from flatten_model import flatten_model
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
from rebuild_model import rebuild_model

device = torch.device('cuda:2')
save_epoch = 200
train_epoch = 50

model = VGG('Small', 10).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones = [10, 20, 30], gamma = 0.2)
criterion = torch.nn.CrossEntropyLoss().to(device)
train_layer = [name for name in model.state_dict()]

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
train_transform = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                            (0.2023, 0.1994, 0.2010)),
    ]
)
train_dataset = torchvision.datasets.CIFAR10(root="./data",train=True,transform=train_transform,download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size = 32, num_workers=4, shuffle=True)
test_dataset = torchvision.datasets.CIFAR10(root="./data",train=False,transform=test_transform,download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size = 32, num_workers=4, shuffle=False)

for epoch in range(train_epoch):
    total_loss = []
    model.train()
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        out = model(data)
        loss = criterion(out, target)
        total_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'In Epoch:{epoch}/{train_epoch}, Mean Loss is:{sum(total_loss)/len(total_loss)}')
    scheduler.step()
    acc, test_loss, _ = evaluate_model(model, test_loader, device)
    print(f'Test Acc is:{acc}, Test Loss is:{test_loss}')

flattened_params = []
for epoch in range(save_epoch):
    model.train()
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        out = model(data)
        loss = criterion(out, target)
        total_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    acc, test_loss, _ = evaluate_model(model, test_loader, device)
    print(f'In Save Epoch, Test Acc is:{acc}, Test Loss is:{test_loss}')
    model.eval()
    flattened = flatten_model(model, train_layer).detach().cpu()
    flattened_params.append(flattened)
    
flattened_params = torch.stack(flattened_params)

# for index in range(5):
#     template_model = VGG('Small', 10).to(device)
#     now_param = flattened_params[index].to(device)
#     now_model = rebuild_model(now_param, template_model, train_layer)
#     acc, test_loss, _ = evaluate_model(now_model, test_loader, device)
#     print(f'In Rebuild Round, Acc:{acc}, Test Loss:{test_loss}')


save_info = {'pdata':flattened_params}
torch.save(save_info, './param_data/vgg-cifar10.pt')


