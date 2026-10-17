import csv
import os
import sys
import time

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

PROJ = r'E:\LinJY\desktop\mnist_from_scratch'
DATA_ROOT = PROJ + r'\data\mnist'
OUT = 'outputs'

sys.path.insert(0, PROJ)
from model import Model  # noqa: E402

MEAN, STD = 0.1307, 0.3081
BATCH, EPOCHS, VAL_SIZE, SEED = 64, 5, 5000, 42

CONFIGS = [
    {'note': 'sgd_p0.3_lr0.01', 'opt': 'sgd', 'lr': 0.01, 'p': 0.3, 'seed': SEED},
    {'note': 'rmsprop_p0.3', 'opt': 'rmsprop', 'lr': 1e-3, 'p': 0.3, 'seed': SEED},
    {'note': 'adam_p0.2', 'opt': 'adam', 'lr': 1e-3, 'p': 0.2, 'seed': SEED},
    {'note': 'adam_p0.5', 'opt': 'adam', 'lr': 1e-3, 'p': 0.5, 'seed': SEED},
    {'note': 'adam_p0.3_s1', 'opt': 'adam', 'lr': 1e-3, 'p': 0.3, 'seed': 1},
    {'note': 'adam_p0.3_s2', 'opt': 'adam', 'lr': 1e-3, 'p': 0.3, 'seed': 2},
    {'note': 'adam_p0.3_s3', 'opt': 'adam', 'lr': 1e-3, 'p': 0.3, 'seed': 3},
]


def make_loaders():
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
    return (DataLoader(Subset(train_full, order[VAL_SIZE:]), batch_size=BATCH, shuffle=True, num_workers=0),
            DataLoader(Subset(val_full, order[:VAL_SIZE]), batch_size=512, shuffle=False, num_workers=0))


def make_optimizer(name, params, lr):
    if name == 'adam':
        return torch.optim.Adam(params, lr=lr)
    if name == 'sgd':
        return torch.optim.SGD(params, lr=lr, momentum=0.9)
    if name == 'rmsprop':
        return torch.optim.RMSprop(params, lr=lr)
    raise ValueError(name)


def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.inference_mode():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item()
            total += y.size(0)
    return correct / total


def run(cfg, trainloader, valloader, device):
    print(f"=== {cfg['note']}  {cfg['opt']} lr={cfg['lr']} p={cfg['p']} seed={cfg['seed']} ===")
    torch.manual_seed(cfg['seed'])
    model = Model(p=cfg['p']).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = make_optimizer(cfg['opt'], model.parameters(), cfg['lr'])

    best, best_epoch, history = 0.0, 0, []
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

        val_acc = evaluate(model, valloader, device)
        train_loss, train_acc = loss_sum / total, correct / total
        print(f"  epoch {epoch} | loss {train_loss:.4f} | train {train_acc:.2%} | val {val_acc:.2%}")
        history.append([epoch, f'{train_loss:.4f}', f'{train_acc:.4f}', f'{val_acc:.4f}'])
        if val_acc > best:
            best, best_epoch = val_acc, epoch

    with open(os.path.join(OUT, f"exp_{cfg['note']}.csv"), 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['epoch', 'train_loss', 'train_acc', 'val_acc'])
        w.writerows(history)
    print(f"  best {best:.2%} @ epoch {best_epoch}\n")
    return best, best_epoch


def main():
    os.makedirs(OUT, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('设备', device)
    trainloader, valloader = make_loaders()

    rows = []
    for cfg in CONFIGS:
        t0 = time.time()
        best, ep = run(cfg, trainloader, valloader, device)
        rows.append([time.strftime('%H:%M'), cfg['note'], cfg['opt'], cfg['lr'], cfg['p'],
                     cfg['seed'], f'{best:.4f}', ep, f'{time.time() - t0:.0f}s'])

    with open(os.path.join(OUT, 'sweep.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['time', 'note', 'opt', 'lr', 'p', 'seed', 'best_val', 'peak_ep', '耗时'])
        w.writerows(rows)

    print('=' * 62)
    print(f"{'note':<20}{'opt':>8}{'lr':>8}{'p':>5}{'best_val':>10}{'peak':>5}")
    for _, note, opt, lr, p, seed, bv, ep, _t in sorted(rows, key=lambda r: -float(r[6])):
        print(f'{note:<20}{opt:>8}{lr:>8}{p:>5}{float(bv):>9.2%}{ep:>5}')


if __name__ == '__main__':
    main()
