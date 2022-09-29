import torch.nn as nn
import torch

class QuantizerSd:
    """
        QuantizerSd quantizes a network's weights for LSTM and Linear layers
        to support Quantization Aware Training. It does so by making a copy of 
        state_dict, changing the tensor weights with quantized/dequantized 
        weights and loading it back to the network. 
    """
    def __init__(self):
        self.helper_dict = {}

    def quantize(self, model: nn.Module):
        """
        """

        # dictionary of weights
        d = model.state_dict()
        for key in d.keys():
            d[key] = self._quantize(d[key], key)

        model.load_state_dict(d)
            
    def dequantize(self, model: nn.Module):

        # dictionary of weights
        d = model.state_dict()
        for key in d.keys():
            d[key] = self._dequantize(d[key], key)
        model.load_state_dict(d)
      

    def _quantize(self, x: torch.tensor, key_name: str, num_bits: int=8):
        """
        Apply quantization, convert data type to uint8 and return 
        the tensor 
        """
        # t.apply_(lambda i: round((i/self.scale) + self.zero_point))
        # t.type(torch.uint8)
        # t = t.to(torch.int8)
        # print(t)
        # t = torch.quantize_per_tensor(t, 0.1, 10, torch.qint8)
        qmin = 0.
        qmax = 2.**num_bits - 1.

        min_val, max_val = x.min(), x.max()

        scale = (max_val - min_val) / (qmax - qmin)

        initial_zero_point = qmin - min_val / scale

        zero_point = 0
        if initial_zero_point < qmin:
            zero_point = qmin
        elif initial_zero_point > qmax:
            zero_point = qmax
        else:
            zero_point = initial_zero_point

        zero_point = int(zero_point)

        q_x = zero_point + x / scale

        q_x.clamp_(qmin, qmax).round_()
        q_x = q_x.round().byte()
        # save scale and zero point for this tensor 
        self.helper_dict[key_name] = tuple((scale, zero_point))

        return q_x
    
    def _dequantize(self, x: torch.tensor, key_name: str):
        """
        Convert the data type to float32 apply Dequantization and return 
        the tensor 
        """
        scale, zero_point = self.helper_dict[key_name]
        x = scale * (x.float() - zero_point)
        x.requires_grad_(True)
        return x 


if __name__ == "__main__":
    m = nn.LSTM(1, 1)
    print("Before")
    print(m.state_dict())
    q = QuantizerSd()

    q.quantize(m)
    print("Quantization")
    print(m.state_dict())

    q.dequantize(m)
    print("Dequantization")
    print(m.state_dict())




