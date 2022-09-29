import torch.nn as nn
import torch

class QuantizerTr:

    """
        QuantizerTr quantizes a network's weights for LSTM and Linear layers
        to support Quantization Aware Training. It does so by traversing through
        the layers and changing the tensor weights with quantized/dequantized 
        weights in-place. 
    """
    def __init__(self, scale=0.1, zero_point=10):
        self.scale = scale
        self.zero_point = zero_point 
        self.dtype = torch.qint8

    def quantize(self, model: nn.Module):
        """
        """
        for layer in model.children():
            if(isinstance(layer, nn.LSTM) or isinstance(layer, nn.Linear)):
                for param in layer.parameters():
                    param.detach().apply_(lambda i: round((i / self.scale) + self.zero_point))
                    param.requires_grad_(True)
            else:
                continue
        
    def dequantize(self, model: nn.Module):

        for layer in model.children():
            if(isinstance(layer, nn.LSTM) or isinstance(layer, nn.Linear)):
                for param in layer.parameters():
                    param.detach().apply_(lambda i: (i - self.zero_point) * self.scale)
                    param.requires_grad_(True)
            else:
                continue



if __name__ == "__main__":
    m = nn.Sequential(nn.LSTM(1, 1), nn.Linear(1, 1))
    # print("Before")
    print(m.state_dict())
    q = QuantizerTr(0.23, 11)

    q.quantize(m)
    print("Quantization")
    print(m.state_dict())
    q.dequantize(m)
    print("Dequantization")
    print(m.state_dict())

    # for layer in m.children():
    #     for param in layer.parameters():
    #         print(param.dtype)




