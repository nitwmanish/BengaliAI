import pretrainedmodels
import torch.nn as nn
from torch.nn import functional as F

class ResNet34(nn.Module):
	def __init__(self, pretrained):
		super(ResNet34, self).__init__()
		if pretrained is True:
			self.model = pretrainedmodels.__dict__["restnet34"](pretrained="imagenet")
		else:
			self.model = pretrainedmodels.__dict__["restnet34"](pretrained=None)
			
		self.l0 = nn.Linear(512, 168)
		self.l1 = nn.Linear(512, 11)
		self.l2 = nn.Linear(512, 7)
		
	def forword(self, x):
		bs, channel, height, width = x.shape
		x = self.model.feature(x)
		x = adaptive_avg_pool2d(x, 1).reshape(bs, -1)
		l0 = self.l0(x)
		l1 = self.l1(x)
		l2 = self.l2(x)
		return l0, l1, l2