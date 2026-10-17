import csv
import glob
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = 'outputs'


def read_rows(path):
    with open(path, encoding='utf-8') as f:
        return list(csv.DictReader(f))


def plot_single():
    path = os.path.join(OUT, 'history.csv')
    if not os.path.exists(path):
        return None
    rows = read_rows(path)
    epochs = [int(r['epoch']) for r in rows]
    train_acc = [float(r['train_acc']) for r in rows]
    val_acc = [float(r['val_acc']) for r in rows]
    train_loss = [float(r['train_loss']) for r in rows]
    peak = max(range(len(val_acc)), key=lambda i: val_acc[i])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    axes[0].plot(epochs, train_acc, 'o-', label='train')
    axes[0].plot(epochs, val_acc, 's-', label='val')
    axes[0].axvline(epochs[peak], color='crimson', ls='--', lw=1)
    axes[0].annotate(f'peak {val_acc[peak]:.2%} @ epoch {epochs[peak]}',
                     xy=(epochs[peak], val_acc[peak]),
                     xytext=(-140, 25), textcoords='offset points',
                     fontsize=9, color='crimson')
    axes[0].set_xlabel('epoch')
    axes[0].set_ylabel('accuracy')
    axes[0].set_title('accuracy')
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, train_loss, 'o-', color='tab:red')
    axes[1].set_xlabel('epoch')
    axes[1].set_ylabel('train loss')
    axes[1].set_title('train loss')
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'curve.png'), dpi=160)
    plt.close(fig)
    return f'peak val {val_acc[peak]:.2%} @ epoch {epochs[peak]}'


def plot_compare():
    files = sorted(glob.glob(os.path.join(OUT, 'exp_*.csv')))
    if not files:
        return None

    fig, ax = plt.subplots(figsize=(8, 5))
    summary = []
    for path in files:
        name = os.path.basename(path)[4:-4]
        rows = read_rows(path)
        epochs = [int(r['epoch']) for r in rows]
        val = [float(r['val_acc']) for r in rows]
        best = max(val)
        summary.append((best, name))
        ax.plot(epochs, val, 'o-', label=f'{name}  best {best:.2%}')

    ax.set_xlabel('epoch')
    ax.set_ylabel('val accuracy')
    ax.set_title('val accuracy comparison')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'compare.png'), dpi=160)
    plt.close(fig)

    summary.sort(reverse=True)
    return summary


if __name__ == '__main__':
    msg = plot_single()
    print(msg or 'history.csv 不在，跳过单次曲线')
    summary = plot_compare()
    if summary:
        print('多组对比（按 val 排序）:')
        for best, name in summary:
            print(f'  {name:<16}{best:.2%}')
        print(f'  已存 {OUT}/compare.png')
    else:
        print('没有 exp_*.csv，跳过多组对比')
