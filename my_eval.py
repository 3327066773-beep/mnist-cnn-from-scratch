import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import Model

DATA_ROOT = 'data/mnist'
OUT = 'outputs'

ckpt = torch.load(f'{OUT}/model.pt', map_location='cpu', weights_only=True)
MEAN, STD = ckpt['mean'], ckpt['std']

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((MEAN,), (STD,)),
])
testset = datasets.MNIST(root=DATA_ROOT, train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=512, shuffle=False, num_workers=0)

model = Model()
model.load_state_dict(ckpt['model_state'])
model.eval()

ys, preds, confs, imgs = [], [], [], []
with torch.inference_mode():
    for x, y in testloader:
        p = model(x).softmax(1)
        ys.append(y)
        preds.append(p.argmax(1))
        confs.append(p.max(1).values)
        imgs.append(x)

y_true = torch.cat(ys).numpy()
y_pred = torch.cat(preds).numpy()
conf = torch.cat(confs).numpy()
images = torch.cat(imgs) * STD + MEAN          # 反归一化，图才看得清

acc = float((y_true == y_pred).mean())
print(f'测试准确率 {acc:.2%}')
print(f'错的总数 {int((y_true != y_pred).sum())} / {len(y_true)}')
print()
print(classification_report(y_true, y_pred, digits=4, zero_division=0))

cm = confusion_matrix(y_true, y_pred)
fig, ax = plt.subplots(figsize=(8, 8))
ConfusionMatrixDisplay(cm).plot(ax=ax, cmap='Blues', colorbar=False, values_format='d')
ax.set_title(f'MNIST test accuracy = {acc:.2%}')
fig.tight_layout()
fig.savefig(f'{OUT}/confusion_matrix.png', dpi=160)
plt.close(fig)

print('最容易混的 8 对（真值 -> 误判成）:')
pairs = []
for i in range(10):
    for j in range(10):
        if i != j and cm[i, j] > 0:
            pairs.append((cm[i, j], i, j))
pairs.sort(reverse=True)
for n, i, j in pairs[:8]:
    print(f'  {i} -> {j}   {n} 次')

wrong = np.flatnonzero(y_true != y_pred)[:25]
cols, rows = 5, (len(wrong) + 4) // 5
fig, axes = plt.subplots(rows, cols, figsize=(10, 2 * rows))
for ax in np.ravel(axes):
    ax.axis('off')
for ax, i in zip(np.ravel(axes), wrong):
    ax.imshow(images[i, 0].numpy(), cmap='gray')
    ax.set_title(f'true={y_true[i]} pred={y_pred[i]}\nconf={conf[i]:.2f}', fontsize=9)
fig.tight_layout()
fig.savefig(f'{OUT}/misclassified.png', dpi=160)
plt.close(fig)
print()
print('两张图已存:', f'{OUT}/confusion_matrix.png', f'{OUT}/misclassified.png')
