import csv
import os

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from model import Model

DATA_ROOT = 'data/mnist'
OUT = 'outputs'
MEAN, STD = 0.1307, 0.3081
BATCH, VAL_SIZE, SEED, EPOCHS = 64, 5000, 42, 10
LR, P = 1e-3, 0.3

tf_train = transforms.Compose([
    transforms.RandomAffine(degrees=8, translate=(0.06, 0.06)),
    transforms.ToTensor(),
    transforms.Normalize((MEAN,), (STD,)),
])
tf_eval = transforms.Compose([transforms.ToTensor(), transforms.Normalize((MEAN,), (STD,))])

train_full = datasets.MNIST(root=DATA_ROOT, train=True, download=True, transform=tf_train)
val_full = datasets.MNIST(root=DATA_ROOT, train=True, download=True, transform=tf_eval)
g = torch.Generator().manual_seed(SEED)
order = torch.randperm(len(train_full), generator=g).tolist()
trainloader = DataLoader(Subset(train_full, order[VAL_SIZE:]), batch_size=BATCH, shuffle=True, num_workers=0)
valloader = DataLoader(Subset(val_full, order[:VAL_SIZE]), batch_size=512, shuffle=False, num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(SEED)
model = Model(p=P).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

print('设备', device, ' epochs', EPOCHS)
history = []
best, best_epoch = 0.0, 0
for epoch in range(1, EPOCHS + 1):
    model.train()
    loss_sum = correct = total = 0
    for x, y in trainloader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * y.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)

    model.eval()
    v_correct = v_total = 0
    with torch.inference_mode():
        for x, y in valloader:
            x, y = x.to(device), y.to(device)
            v_correct += (model(x).argmax(1) == y).sum().item()
            v_total += y.size(0)
    val_acc = v_correct / v_total
    train_loss, train_acc = loss_sum / total, correct / total
    print(f'epoch {epoch:2d} | loss {train_loss:.4f} | train {train_acc:.2%} | val {val_acc:.2%}')
    history.append([epoch, f'{train_loss:.4f}', f'{train_acc:.4f}', f'{val_acc:.4f}'])
    if val_acc > best:
        best, best_epoch = val_acc, epoch

with open(os.path.join(OUT, 'exp_adam_p0.3_aug_ep10.csv'), 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f)
    w.writerow(['epoch', 'train_loss', 'train_acc', 'val_acc'])
    w.writerows(history)
print(f'best {best:.2%} @ epoch {best_epoch}')
