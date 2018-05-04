import torch
import torch.nn as nn

class STNLayer(nn.Module):
    def __init__(self, channel, joint=False):
        super(STNLayer, self).__init__()
	# Start
	self.conv_1x1 = nn.Conv2d(channel, channel/32, kernel_size=1, stride=1, bias=False)
	
	if channel == 256:
	    kernel1 = 7
	    kernel2 = 7 # 5
	elif channel == 512:
	    kernel1 = 5
	    kernel2 = 3
	else: # 1024
	    kernel1 = 3
	    kernel2 = 1

	# Encoder
        self.en_conv1 = nn.Conv2d(channel/32, channel/8, kernel_size=kernel1)
        self.en_act1 = nn.ReLU(True)
        self.en_pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, return_indices=True)
	self.en_conv2 = nn.Conv2d(channel/8, channel/4, kernel_size=kernel2)

	# Decoder
	self.de_conv1 = nn.ConvTranspose2d(channel/4, channel/8, kernel_size=kernel2)
	self.de_act1 = nn.ReLU(True)
        self.de_pool1 = nn.MaxUnpool2d(kernel_size=2, stride=2) 
	self.de_conv2 = nn.ConvTranspose2d(channel/8, channel/32, kernel_size=kernel1)
        
	# Final
	self.conv_1x1_2 = nn.Conv2d(channel/32, 1, kernel_size=1, stride=1)

	if joint==True:
	    self.joint = 1
	else:
	    self.joint = 0

    def forward(self, x):
	y = self.conv_1x1(x)

	y1 = self.en_conv1(y)
	y, indices1 = self.en_pool1(y1)
	y = self.en_act1(y)
	y = self.en_conv2(y)
	
	y = self.de_conv1(y)
	y = self.de_act1(y)
        y = self.de_pool1(y, indices1, output_size=y1.size())
	y = self.de_conv2(y)

	y = self.conv_1x1_2(y)

	if self.joint == 0: # stn module
	    return x * y
	else:		    # joint module
	    return y

