import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# References:
# N: mini-batch size
# C: number of channes (or feature maps)
# H: vertical dimension of feature map (or input image)
# W: horizontal dimension of feature map (or input image)
# K: number of different classes (unique chars)

class HTRModel(nn.Module):
    def __init__(self, num_classes, line_height=64):
        super(HTRModel, self).__init__()
        
        # https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html
        self.layer1 = nn.Sequential(
            # https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
            # Conv2d(in_channels, out_channels, kernel_size, stride=1, 
            #        padding=0, dilation=1, groups=1, bias=True, 
            #        padding_mode='zeros', device=None, dtype=None)
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            #
            # https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html
            # torch.nn.ReLU(inplace=False)
            nn.ReLU(),
            # 
            # https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html
            # MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, 
            #           return_indices=False, ceil_mode=False)
            nn.MaxPool2d((2,1)))

        # Conv feature-map output height: line_height x 64 / 2 
        # ---> 1024 if line_height=32
        lstmInpSz = int(line_height * 64 / 2) 
        #lstmInpSz = int(line_height * 128 / 2) 

        # https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
        # LSTM(input_size, hidden_size, num_layers=1, bias=True, 
        #      batch_first=False, dropout=0, bidirectional=False, proj_size=0)
        self.lstm = nn.LSTM(lstmInpSz, 100, bidirectional=True)

        # https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html
        # Dropout(p=0.5, inplace=False)
        #self.dropout = nn.Dropout(p=0.5)
        
        # https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
        # Linear(in_features, out_features, bias=True)
        self.fc = nn.Linear(100*2, num_classes)



    def forward(self, x):
        # x ---> Tensor:NxCxHxW, C=1 for binary/grey-scale images

        x = self.layer1(x)
        # x ---> Tensor:NxCxH'xW, H'=H/2 because of maxpooling (2x1)

        x = x.permute(0, 3, 1, 2)
        # x ---> Tensor:NxWxCxH'
        
        x = x.view(x.size(0), x.size(1), -1)
        # x ---> Tensor:NxWxC*H' 
        
        x = x.permute(1, 0, 2)
        # x ---> Tensor:WxNxC*H', because of batch_first=False was set by
        #                         default (see: https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html)
        
        x, _ = self.lstm(x)
        # x ---> Tensor:WxNxD*100, D=2 as bidireccional=True was set
        #                          100 is the number of LSTM units
        
        #x = self.dropout(x)
        # x ---> Tensor:WxNxD*100, D=2 as bidireccional=True was set
        #                          100 is the number of LSTM units

        x = self.fc(x)
        # x ---> Tensor:WxNxK, K=number of different char classes + 1

        # https://pytorch.org/docs/stable/generated/torch.nn.functional.log_softmax.html
        # log_softmax(input, dim=None)
        return F.log_softmax(x, dim=2)

