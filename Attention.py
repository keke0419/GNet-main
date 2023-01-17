import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import init

class ExternalAttention(nn.Module):

    def __init__(self, d_model,S=64):
        super().__init__()
        self.mk=nn.Linear(S,S,bias=False)
        self.mv=nn.Linear(S,d_model,bias=False)
        self.softmax=nn.Softmax(dim=1)
        self.init_weights()


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, queries):
        attn=self.mk(queries) #bs,n,S
        attn=self.softmax(attn) #bs,n,S
        attn=attn/torch.sum(attn,dim=2,keepdim=True) #bs,n,S
        out=self.mv(attn) #bs,n,d_model

        return out

class DualAttention(nn.Module):

    def __init__(self, d_in,d_out):
        super().__init__()
        self.mk=nn.Linear(d_out,d_in,bias=False)
        self.mv=nn.Linear(d_in,d_out,bias=False)
        self.softmax=nn.Softmax(dim=1)
        self.init_weights()
        # self.gamma = nn.Parameter(torch.randn((1, 1, 1)))
        # self.PosEncod1 = PositionalEncoding(64, dropout=0, max_len=15)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, input):
        attn = torch.matmul(input.permute(0, 2, 1), input) #bs,n,S
        attn = self.softmax(attn)  # bs,n,S
        attn = attn/torch.sum(attn,dim=2,keepdim=True) #bs,n,S
        out = torch.matmul(input, attn)
        out = self.mv(out) #bs,n,d_model
        return out