import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, densenet121, mobilenet_v2

class LeNet5(nn.Module):
    def __init__(self, num_classes, in_channels=1, use_cuda=True):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 6, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        self.use_cuda = use_cuda
        if use_cuda and torch.cuda.is_available():
            self.cuda()

    def forward(self, x):
        if self.use_cuda and torch.cuda.is_available():
            x = x.cuda()
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def create_model(model_name, num_classes, in_channels=1, use_cuda=True):
    """
    创建一个模型。

    Args:
        model_name (str): 模型名称（resnet18, resnet18_grayscale, densenet121, mobilenetv2, lenet5）。
        num_classes (int): 类别数。
        in_channels (int, optional): 输入通道数，仅适用于 LeNet5。默认为 1。
        pretrained (bool, optional): 是否使用预训练模型。默认为 False。
        use_cuda (bool, optional): 是否将模型放置到 GPU 上。默认为 True。

    Returns:
        nn.Module: 创建的模型。

    Raises:
        ValueError: 如果模型名称无效。
    """

    if model_name == "resnet18":
        model = resnet18(num_classes=num_classes, weights=None)

    elif model_name == "densenet121":
        model = densenet121(weights=None)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, num_classes)

    elif model_name == "mobilenetv2":
        model = mobilenet_v2(weights=None)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)

    elif model_name == "lenet5":
        model = LeNet5(num_classes=num_classes, in_channels=in_channels, use_cuda=use_cuda)

    else:
        raise ValueError("Invalid model name.")

    if use_cuda and torch.cuda.is_available():
        model = model.cuda()

    return model
#