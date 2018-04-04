from .. import node, np, Variable

class Conv1d(node):
	def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
		pass

	def forward(self, x):
		pass

class Conv2d(node):
	def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
		pass
	def forward(self, x):
		y = F.convolution_2d(x, W, b, stride=(s_y, s_x), pad=(h_p, w_p), cover_all=True)