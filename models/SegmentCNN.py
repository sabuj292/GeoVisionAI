import torch.nn as nn
import torchvision.models as models

class ConvBlock(nn.Module):
    def __init__(self, in_chn, out_chn, kernel_size = 3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_chn, out_chn, kernel_size=kernel_size, padding = 'same')
        self.dropout1 = nn.Dropout(0.2)
        self.conv2 = nn.Conv2d(out_chn, out_chn, kernel_size=kernel_size, padding = 'same')
        
    def forward(self, x):
        x = nn.ReLU()(self.conv1(x))
        x = self.dropout1(x)
        x = nn.ReLU()(self.conv2(x))
        return x
        
class UpScale(nn.Module):
    def __init__(self, in_chn, out_chn, poolsize = 2):
        super().__init__()
        self.convblock = ConvBlock(in_chn, out_chn)
        self.maxpool1 = nn.MaxPool2d(poolsize)
        
    def forward(self, x):
        x = self.convblock(x)
        p = self.maxpool1(x)
        return x, p

class DownScale(nn.Module):
    def __init__(self, in_chn, out_chn, poolsize = 2):
        super().__init__()
        self.transConv1 = nn.ConvTranspose2d(in_chn, out_chn, kernel_size=2, stride=2)
        self.convblock = ConvBlock(in_chn, out_chn)
        
    def forward(self, x1, x2):
        u = self.transConv1(x1)
        u = torch.cat((u, x2), 1)
        x = self.convblock(u)
        return x

class Model(nn.Module):
    def __init__(self, n_classes = 7, channels = 3):
        super().__init__()
        self.upblock1 = UpScale(channels, 4)
        self.upblock2 = UpScale(4, 8)
        self.upblock3 = UpScale(8, 16)
        self.upblock4 = UpScale(16, 32)

        self.transferblock = ConvBlock(32, 64)

        self.downblock1 = DownScale(64, 32)
        self.downblock2 = DownScale(32, 16)
        self.downblock3 = DownScale(16, 8)
        self.downblock4 = DownScale(8, 4)

        self.final_conv = nn.Conv2d(4, n_classes, kernel_size=1, padding = 'same')

    def forward(self, x):
        """ Encoder """
        x1, p1 = self.upblock1(x)
        x2, p2 = self.upblock2(p1)
        x3, p3 = self.upblock3(p2)
        x4, p4 = self.upblock4(p3)

        """Transfer block"""
        x5 = self.transferblock(p4)

        """ Decoder """
        x6 = self.downblock1(x5, x4)
        x7 = self.downblock2(x6, x3)
        x8 = self.downblock3(x7, x2)
        x9 = self.downblock4(x8, x1)

        '''Classifier'''
        output = self.final_conv(x9)
        return output