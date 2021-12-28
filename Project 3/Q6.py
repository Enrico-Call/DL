import torch
import torch.nn.functional as F
from torch import nn


class Conv2D(torch.autograd.Function):

	@staticmethod
	def forward(ctx, inputs, kernel):

		ctx.save_for_backward(inputs, kernel)
		unfold = nn.Unfold(kernel_size=(3, 3))
		u = unfold(inputs)
		yh = u.matmul(kernel)
		y = yh.reshape(-1)

		return y

	@staticmethod
	def backward(ctx, grad_output):

		inputs, kernel = ctx.saved_tensors
		grout = unfold(grad_output)
		input_batch_grad = torch.matmul(inputs, grout)
		kernel_grad = torch.matmul(kernel, grout)
		kernel_grad = torch.fold(kernel_grad)
		
		return input_batch_grad, kernel_grad


torch.manual_seed(0)