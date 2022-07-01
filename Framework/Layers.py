from torch.autograd import Function

class GradientReversal(Function):

    @staticmethod
    def forward(self, x):
        # view_as creates new tensor with shape of argument tensor
        # might be unnecessary
        #print("forward called")
        return x.view_as(x)

    @staticmethod
    def backward(self, grad):
        #print("backward called")
        lambd = 0.5
        return (grad * (-lambd))
