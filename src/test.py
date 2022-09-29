import torch 
from quantizer import * 

m = nn.Sequential(nn.LSTM(1,1), nn.Linear(1, 1))

print("before")
print(m.state_dict())
stats = {}

weights = simulate_quantization(m, stats)

print(stats)

print("after")
print(m.state_dict())