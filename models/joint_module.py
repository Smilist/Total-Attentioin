import torch
import torch.nn as nn

from se_module import SELayer
from stn_module import STNLayer

class JOINTLayer(nn.Module):
    def __init__(self, channel):
        super(JOINTLayer, self).__init__()
	self.SE_layer = SELayer(channel, joint=True)
	self.STN_layer = STNLayer(channel, joint=True)
	self.act = nn.Sigmoid()

    def forward(self, x):
	#print('input: ' + str(x.shape))
	y1 = self.SE_layer(x)  # dimension: c*1*1
	y2 = self.STN_layer(x) # dimension: 1*H*W
	#print('1. SE layer results: ' + str(type(y1)))
	#print('2. STN layer results: ' + str(type(y2)))
	
	#y = torch.add(y1+y2)  # y = y1 + y2
	y = y1+y2
	#print('3. add results: ' + str(y.shape))

	y = self.act(y)
	#print('4. activation results: ' + str(y.shape))

	attention_out = x*y
	#print('attention output: ' + str(attention_out.shape))

	out = torch.add(attention_out, x)
	#print('real output: ' + str(out.shape))
        return out 
