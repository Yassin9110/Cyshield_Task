# src/models/grl.py

import torch
from torch.autograd import Function

class GradientReversalFunction(Function):
    """
    Gradient Reversal Layer from:
    Unsupervised Domain Adaptation by Backpropagation (Ganin & Lempitsky, 2015)
    """
    @staticmethod
    def forward(ctx, x, lambda_val):
        # Store lambda_val for the backward pass
        ctx.lambda_val = lambda_val
        # Forward pass is the identity function
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        # Backward pass reverses the gradient and scales it by lambda
        # The gradient w.r.t. lambda is None
        return (grad_output.neg() * ctx.lambda_val), None

class GradientReversalLayer(torch.nn.Module):
    def __init__(self):
        super(GradientReversalLayer, self).__init__()

    def forward(self, x, lambda_val=1.0):
        return GradientReversalFunction.apply(x, lambda_val)