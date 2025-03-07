import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

def choose_dataset(name, batch_size):
    """选择数据集并返回 DataLoader 和类别数目."""
    if name == 'STL10':
        # STL10 的特殊处理
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4467, 0.4398, 0.4066), (0.2603, 0.2565, 0.2712))
        ])
        train_dataset = datasets.STL10(root='./data', split='train', transform=transform, download=True)
        test_dataset = datasets.STL10(root='./data', split='test', transform=transform, download=True)
    else:
        transform = {
            'MNIST': transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]),
            'FashionMNIST': transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]),
            'CIFAR10': transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
            'CIFAR100': transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        }[name]

        train_dataset = datasets.__dict__[name](root='./data', train=True, transform=transform, download=True)
        test_dataset = datasets.__dict__[name](root='./data', train=False, transform=transform, download=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    num_classes = len(train_dataset.classes)

    return train_dataset, test_dataset, train_loader, test_loader, num_classes


