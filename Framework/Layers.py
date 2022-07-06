from torch.autograd import Function

class GradientReversal(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        # view_as creates new tensor with shape of argument tensor
        # might be unnecessary
        #print("forward called")
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        #print("backward called")
        return (grad * (-ctx.alpha)), None
