from torch import nn

class SELayer(nn.Module):
    def __init__(self, channel, joint=False):
        super(SELayer, self).__init__()
	reduction = 16

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),
                nn.Sigmoid()
        )

        if joint==True:
            self.joint = 1
        else:
            self.joint = 0

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)

	if self.joint == 0: # se module
	    return x * y
	else: 		    # joint module
            return y
