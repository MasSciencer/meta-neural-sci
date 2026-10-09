import torch

class SigmoidDecoder(torch.nn.Module):
    def __init__(self, pmin=0.1, pmax=100.0):
        super().__init__()
        self.pmin = pmin
        self.pmax = pmax

    def forward(self, u):
        s = torch.sigmoid(u)
        return self.pmin + (self.pmax - self.pmin) * s

###########################################################################
# class Decoder(torch.nn.Module):
#     def __init__(self, temp=1.0, scale=0.5):
#         super().__init__()
#         self.temp = temp
#         self.scale = scale
#
#     def forward(self, u):
#         # Exponential growth with temperature and scale parameters.
#         v = u / self.temp
#         p = torch.exp(v) * self.scale
#         # Clamp the result to avoid unbounded growth.
#         return torch.clamp(p, max=100)
#
# class Decoder(torch.nn.Module):
#     def __init__(self, beta = 1.0 ,threshold=20):
#         super().__init__()
#         self.threshold = threshold
#         self.beta = beta
#
#     def forward(self, u):
#         v = torch.clamp(u, min=0.001, max=100.0)
#         p = F.softplus(v, beta=self.beta, threshold=self.threshold)
#         return p