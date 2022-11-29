from torch.autograd import Function

class GradientReversal(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        # view_as creates new tensor with shape of argument tensor
        # might be unnecessary
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return (grad * (-ctx.alpha)), None
