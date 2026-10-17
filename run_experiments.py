import csv
import os

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

import experiment
from model import Model

DATA_ROOT = 'data/mnist'
OUT = 'outputs'
MEAN, STD = 0.1307, 0.3081
BATCH, EPOCHS, LR, VAL_SIZE, SEED = 64, 5, 1e-3, 5000, 42

# 想试什么就改这个列表：一堆配置连续跑，每个自动记一行
CONFIGS = [
    {'note': 'p0.0_aug', 'p': 0.0, 'aug': 1, 'batch': BATCH, 'lr': LR, 'epochs': EPOCHS},
    {'note': 'p0.3_aug', 'p': 0.3, 'aug': 1, 'batch': BATCH, 'lr': LR, 'epochs': EPOCHS},
]


def make_loaders(use_aug):
    ops = []
    if use_aug:
        ops.append(transforms.RandomAffine(degrees=8, translate=(0.06, 0.06)))
    ops += [transforms.ToTensor(), transforms.Normalize((MEAN,), (STD,))]
    tf_train = transforms.Compose(ops)
    tf_eval = transforms.Compose([transforms.ToTensor(), transforms.Normalize((MEAN,), (STD,))])

    train_full = datasets.MNIST(root=DATA_ROOT, train=True, download=True, transform=tf_train)
    val_full = datasets.MNIST(root=DATA_ROOT, train=True, download=True, transform=tf_eval)

    g = torch.Generator().manual_seed(SEED)
    order = torch.randperm(len(train_full), generator=g).tolist()
    trainset = Subset(train_full, order[VAL_SIZE:])
    valset = Subset(val_full, order[:VAL_SIZE])

    return (DataLoader(trainset, batch_size=BATCH, shuffle=True, num_workers=0),
            DataLoader(valset, batch_size=512, shuffle=False, num_workers=0))


def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.inference_mode():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item()
            total += y.size(0)
    return correct / total


def run(cfg, device):
    print(f"=== {cfg['note']}  dropout p={cfg['p']}  aug={cfg['aug']} ===")
    trainloader, valloader = make_loaders(cfg['aug'])

    torch.manual_seed(SEED)                 # 固定初始化，各组配置才可比
    model = Model(p=cfg['p']).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'])

    best, best_epoch, best_path, history = 0.0, 0, None, []

    for epoch in range(1, cfg['epochs'] + 1):
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

        val_acc = evaluate(model, valloader, device)
        train_loss, train_acc = loss_sum / total, correct / total
        print(f"  epoch {epoch} | loss {train_loss:.4f} | train {train_acc:.2%} | val {val_acc:.2%}")
        history.append([epoch, f'{train_loss:.4f}', f'{train_acc:.4f}', f'{val_acc:.4f}'])

        if val_acc > best:
            best, best_epoch = val_acc, epoch
            best_path = experiment.save_peak(model, cfg, val_acc, epoch, MEAN, STD)

    path = os.path.join(OUT, f"exp_{cfg['note']}.csv")
    os.makedirs(OUT, exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['epoch', 'train_loss', 'train_acc', 'val_acc'])
        w.writerows(history)

    experiment.record(cfg, best, best_epoch)
    print(f"  best {best:.2%} @ epoch {best_epoch}\n")
    return best, best_path


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('设备', device)

    results = []
    for cfg in CONFIGS:
        best, path = run(cfg, device)
        results.append((best, path, cfg['note']))

    results.sort(reverse=True)
    best_val, best_path, note = results[0]
    print(f'全局最好: {note}  val {best_val:.2%}')
    if best_path:
        experiment.promote(best_path)
    print()
    experiment.show()


if __name__ == '__main__':
    main()
