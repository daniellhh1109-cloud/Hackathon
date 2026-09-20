"""Mathematically identical grouped Conv1d CPU path for very short sequences.

Keeps nn.Conv1d parameters/state_dict. Other convolution layouts/devices use
PyTorch's native path. Floating-point accumulation order can differ slightly.
"""
import torch
from torch import nn
from torch.nn import functional as F


class ShortSequenceConv1d(nn.Conv1d):
    def forward(self,x):
        if (x.device.type != 'cpu' or x.ndim != 3 or self.stride != (1,)
                or self.dilation != (1,) or self.padding_mode != 'zeros'):
            return super().forward(x)
        batch,channels,time=x.shape
        if self.kernel_size == (1,) and self.padding == (0,):
            groups=self.groups
            inputs=channels//groups
            outputs=self.out_channels//groups
            a=x.reshape(batch,groups,inputs,time).permute(1,0,3,2).reshape(groups,batch*time,inputs)
            weight=self.weight.reshape(groups,outputs,inputs)
            result=torch.bmm(a,weight.transpose(1,2))
            result=result.reshape(groups,batch,time,outputs).permute(1,0,3,2).reshape(batch,self.out_channels,time)
        elif self.groups == channels == self.out_channels and isinstance(self.padding,tuple):
            windows=F.pad(x,(self.padding[0],self.padding[0])).unfold(2,self.kernel_size[0],1)
            result=(windows*self.weight[:,0,:][None,:,None,:]).sum(-1)
        else:
            return super().forward(x)
        return result if self.bias is None else result+self.bias[None,:,None]
