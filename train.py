import csv
import os
import time

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from model import Model

DATA_ROOT = 'data/mnist'
OUT = 'outputs'
ARCHIVE = os.path.join(OUT, 'archive')

NOTE = 'baseline 3conv dropout0'

MEAN, STD = 0.1307, 0.3081
BATCH, EPOCHS, LR, VAL_SIZE = 64, 5, 1e-3, 5000

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((MEAN,), (STD,)),
])

full = datasets.MNIST(root=DATA_ROOT, train=True, download=True, transform=transform)

g = torch.Generator().manual_seed(42)
order = torch.randperm(len(full), generator=g).tolist()
trainset = Subset(full, order[VAL_SIZE:])
valset = Subset(full, order[:VAL_SIZE])

trainloader = DataLoader(trainset, batch_size=BATCH, shuffle=True, num_workers=0)
valloader = DataLoader(valset, batch_size=512, shuffle=False, num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Model().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)


def train_one_epoch(loader):
    """跑一个 epoch，返回 (平均 loss, 训练准确率)。"""
    model.train()
    loss_sum = correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        loss_sum += loss.item() * y.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
    return loss_sum / total, correct / total


def evaluate(loader):
    """只考不改，返回准确率。"""
    model.eval()
    correct = total = 0
    with torch.inference_mode():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item()
            total += y.size(0)
    return correct / total


def save_peak(val_acc, epoch):
    """验证分创新高时调用：更新 model.pt，并在 archive 留一份带分数和时间的。"""
    payload = {'model_state': model.state_dict(), 'mean': MEAN, 'std': STD,
               'val_acc': val_acc, 'epoch': epoch, 'note': NOTE}
    torch.save(payload, os.path.join(OUT, 'model.pt'))

    stamp = time.strftime('%m%d-%H%M%S')
    name = f'epoch{epoch}_val{val_acc:.4f}_{stamp}.pt'
    torch.save(payload, os.path.join(ARCHIVE, name))
    print(f'  新高 -> model.pt 更新，archive/{name}')


def write_history(history):
    """每轮四列数据，给 plot_history.py 画图用。"""
    with open(os.path.join(OUT, 'history.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['epoch', 'train_loss', 'train_acc', 'val_acc'])
        w.writerows(history)


def log_experiment(best_val, best_epoch):
    """一次训练追加一行，跨实验对比用。"""
    path = os.path.join(OUT, 'experiments.csv')
    new_file = not os.path.exists(path)
    with open(path, 'a', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(['time', 'note', 'batch', 'lr', 'epochs',
                        'val_size', 'best_val', 'best_epoch'])
        w.writerow([time.strftime('%m-%d %H:%M'), NOTE, BATCH, LR, EPOCHS, VAL_SIZE,
                    f'{best_val:.4f}', best_epoch])


def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(ARCHIVE, exist_ok=True)
    print('设备', device)
    print('参数量', sum(p.numel() for p in model.parameters()))
    print('本次备注:', NOTE)

    best, best_epoch, history = 0.0, 0, []

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(trainloader)
        val_acc = evaluate(valloader)

        print(f'epoch {epoch} | loss {train_loss:.4f} | '
              f'train {train_acc:.2%} | val {val_acc:.2%}')
        history.append([epoch, f'{train_loss:.4f}', f'{train_acc:.4f}', f'{val_acc:.4f}'])

        if val_acc > best:
            best, best_epoch = val_acc, epoch
            save_peak(val_acc, epoch)

    write_history(history)
    log_experiment(best, best_epoch)

    print(f'best val {best:.2%} @ epoch {best_epoch}')
    print('history.csv / experiments.csv 已写入', OUT)


if __name__ == '__main__':
    main()
