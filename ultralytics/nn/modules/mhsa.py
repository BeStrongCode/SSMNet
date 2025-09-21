import torch

from ultralytics.nn.modules import TransformerBlock

x = torch.randn(8,512,40,40)
model = TransformerBlock(512,512,8,2)
x = model(x)
print(x.shape)