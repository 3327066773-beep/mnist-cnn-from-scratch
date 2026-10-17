import csv
import os
import shutil
import time

import torch

OUT = 'outputs'
ARCHIVE = os.path.join(OUT, 'archive')
EXP_PATH = os.path.join(OUT, 'experiments.csv')
FIELDS = ['time', 'note', 'p', 'aug', 'batch', 'lr', 'epochs', 'best_val', 'best_epoch']


def record(cfg, best_val, best_epoch):
    """一次训练追加一行到 experiments.csv。"""
    os.makedirs(OUT, exist_ok=True)
    new_file = not os.path.exists(EXP_PATH)
    with open(EXP_PATH, 'a', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(FIELDS)
        w.writerow([time.strftime('%m-%d %H:%M'), cfg['note'], cfg['p'], cfg['aug'],
                    cfg['batch'], cfg['lr'], cfg['epochs'],
                    f'{best_val:.4f}', best_epoch])


def save_peak(model, cfg, val_acc, epoch, mean, std):
    """验证分创新高时把权重归档，返回归档路径。"""
    os.makedirs(ARCHIVE, exist_ok=True)
    payload = {'model_state': model.state_dict(), 'mean': mean, 'std': std,
               'val_acc': val_acc, 'epoch': epoch, 'note': cfg['note']}
    stamp = time.strftime('%m%d-%H%M%S')
    path = os.path.join(ARCHIVE, f"{cfg['note']}_ep{epoch}_val{val_acc:.4f}_{stamp}.pt")
    torch.save(payload, path)
    return path


def promote(path):
    """把某份归档提升为 outputs/model.pt，供评估脚本使用。"""
    os.makedirs(OUT, exist_ok=True)
    shutil.copyfile(path, os.path.join(OUT, 'model.pt'))
    print('model.pt <-', os.path.basename(path))


def show():
    if not os.path.exists(EXP_PATH):
        print('还没有实验记录')
        return
    with open(EXP_PATH, encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    print(f"{'note':<14}{'p':>5}{'aug':>4}{'ep':>4}{'best_val':>10}{'peak':>6}")
    print('-' * 45)
    for r in sorted(rows, key=lambda r: -float(r['best_val'])):
        print(f"{r['note']:<14}{r.get('p','?'):>5}{r.get('aug','?'):>4}{r['epochs']:>4}"
              f"{float(r['best_val']):>9.2%}{r['best_epoch']:>6}")


if __name__ == '__main__':
    show()
