from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))])

trainset = datasets.MNIST(root='./data/mnist', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

if __name__ == '__main__':
    print(len(trainset))
    x, y = next(iter(trainloader))
    print(x.shape, y.shape)
