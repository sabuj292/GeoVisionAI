import torch
import torch.nn as nn
import torchvision.models as models

class LandslideClassificationBaseline(torch.nn.Module):
    def __init__(self, pretrained_network = None):
        super().__init__()
        # self.masked_output = pretrained_network
        self.conv2d1 = torch.nn.Conv2d(3, 4, kernel_size=3, stride=2)
        self.dropout1 = nn.Dropout(0.2)
        self.conv2d2 = torch.nn.Conv2d(4, 8, kernel_size=3, stride=2)
        self.dropout2 = nn.Dropout(0.2)
        self.conv2d3 = torch.nn.Conv2d(8, 4, kernel_size=3, stride=2)
        self.dropout3 = nn.Dropout(0.2)
        self.conv2d4 = torch.nn.Conv2d(4, 2, kernel_size=3, stride=2)
        self.dropout4 = nn.Dropout(0.2)
        self.dense1 = torch.nn.Linear(1922, 512)
        self.dense2 = torch.nn.Linear(512, 1)

    def forward(self, x):
        # x = self.masked_output(x)
        
        x = self.conv2d1(x)
        x = nn.ReLU()(x)
        x = self.dropout1(x)
        
        x = self.conv2d2(x)
        x = nn.ReLU()(x)
        x = self.dropout2(x)
        
        x = self.conv2d3(x)
        x = nn.ReLU()(x)
        x = self.dropout3(x)
        
        x = self.conv2d4(x)
        x = nn.ReLU()(x)
        x = self.dropout4(x)
        
        x = torch.nn.Flatten()(x)
        x = self.dense1(x)
        x = torch.nn.ReLU()(x)

        x = self.dense2(x)
        x = torch.nn.Sigmoid()(x)
        return x

    def trainig_step(self, batch):
        images, labels = batch
        labels = torch.unsqueeze(labels, 1).type(torch.float32)
        images = torch.permute(images, (0, 3, 1, 2)).type(torch.float32)
        
        preds = torch.round(self(images)).type(torch.float32)
        
        loss = criterion(self(images).type(torch.float32), labels)
        acc = torch.sum(preds == labels)/len(images)
        
        return loss, acc