from torch.autograd import Function

class GradientReversal(Function):

    @staticmethod
    def forward(self, ctx, x):
        # view_as creates new tensor with shape of argument tensor
        # might be unnecessary
        print("forward called")
        return x.view_as(x)

    @staticmethod
    def backward(self, ctx, grad):
        print("backward called")
        lambd = 1
        return (grad.neg() * lambd), None
