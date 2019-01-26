import torch
import torch.nn as nn
import torch.nn.functional as F


class SiameseNet(nn.Module):

	def __init__(self):
		super(SiameseNet, self).__init__();
		self.cnn = nn.Sequential(
			nn.Conv2d(3, 64, 10),
			nn.BatchNorm3d(),
			


			);

	def forward(self, x):
