import torch.nn as nn


class LeNet5(nn.Module):
    """ LeNet-5 (LeCun et al., 1998)

        - For a detailed architecture, refer to the lecture note
        - Freely choose activation functions as you want
        - For subsampling, use max pooling with kernel_size = (2,2)
        - Output should be a logit vector
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1)
        self.pool = nn.MaxPool2d(kernel_size=(2,2))
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=4, stride=1)
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, 10)
        self.active = nn.Tanh()

    def forward(self, img):
        img = self.conv1(img)
        img = self.active(img)
        img = self.pool(img)

        img = self.conv2(img)
        img = self.active(img)
        img = self.pool(img)

        img = self.conv3(img)
        img = self.active(img)

        img = img.view(img.size(0), -1)
        img = self.fc1(img)
        img = self.active(img)

        output = self.fc2(img)
        return output
    

class LeNet5_advance(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1)
        self.batch1 = nn.BatchNorm2d(6)
        self.pool = nn.MaxPool2d(kernel_size=(2,2))
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)
        self.batch2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=4, stride=1)
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, 10)
        self.active = nn.Tanh()
        

    def forward(self, img):
        img = self.conv1(img)
        img = self.batch1(img)
        img = self.active(img)
        img = self.pool(img)

        img = self.conv2(img)
        img = self.batch2(img)
        img = self.active(img)
        img = self.pool(img)

        img = self.conv3(img)
        img = self.active(img)

        img = img.view(img.size(0), -1)
        img = self.fc1(img)
        img = self.active(img)
        
        output = self.fc2(img)
        return output



class CustomMLP(nn.Module):
    """ Your custom MLP model

        - Note that the number of model parameters should be about the same
          with LeNet-5
    """

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 50)
        self.fc2 = nn.Linear(50, 60)
        self.fc3 = nn.Linear(60, 30)
        self.fc4 = nn.Linear(30, 10)
        self.active = nn.Tanh()
    def forward(self, img):
        img = img.view(img.size(0), -1)

        img = self.fc1(img)
        img = self.active(img)

        img = self.fc2(img)
        img = self.active(img)

        img = self.fc3(img)
        img = self.active(img)

        output = self.fc4(img)
        return output
    

