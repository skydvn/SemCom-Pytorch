import torch.nn as nn
import numpy as np
import os
import torch
import time


class Channel(nn.Module):
    """
    Currently the channel model is either error free, erasure channel,
    rayleigh channel or the AWGN channel.
    """

    def __init__(self, channel_type = 'AWGN', snr = 20):
        super(Channel, self).__init__()
        self.channel_type = channel_type
        self.h = torch.sqrt(torch.randn(1) ** 2
                            + torch.randn(1) ** 2) / 1.414 # tạo hệ số  kênh h với công suất = 1 (sqrt(x^2+y^2)/sqrt(2) = 1)

    def gaussian_noise_layer(self, input_layer, std, name=None):
        
        device = input_layer.get_device()
        noise_real = torch.normal(mean=0.0, std=std, size=np.shape(input_layer), device=device)
        noise_imag = torch.normal(mean=0.0, std=std, size=np.shape(input_layer), device=device)
        noise = noise_real + 1j * noise_imag # tạo noise phức 
        return input_layer + noise

    def rayleigh_noise_layer(self, input_layer, std, name=None):
       # print("Rayleighhhh")
        noise_real = torch.normal(mean=0.0, std=std, size=np.shape(input_layer))
        noise_imag = torch.normal(mean=0.0, std=std, size=np.shape(input_layer))
        noise = noise_real + 1j * noise_imag
        h = torch.sqrt(torch.normal(mean=0.0, std=1, size=np.shape(input_layer)) ** 2
                       + torch.normal(mean=0.0, std=1, size=np.shape(input_layer)) ** 2) / np.sqrt(2)
        # if self.config.CUDA:
        #     noise = noise.to(input_layer.get_device())
        #     h = h.to(input_layer.get_device())
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        noise = noise.to(self.device)
        h = h.to(self.device)
        return input_layer * h + noise


    def complex_normalize(self, x, power):
        pwr = torch.mean(x ** 2) * 2
        out = np.sqrt(power) * x / torch.sqrt(pwr)
        return out, pwr


    def forward(self, input, snr, avg_pwr=False): # input đây là x or feature 
        if avg_pwr: # có chuẩn hóa công suất  = 1 không thì tạo ra channel_tx từ x 
            power = 1
            channel_tx = np.sqrt(power) * input / torch.sqrt(avg_pwr * 2)
        else:
            channel_tx, pwr = self.complex_normalize(input, power=1)
        input_shape = channel_tx.shape

        # bắt đầu bị trải phẳng => B x num_channel x out_dim
        channel_in = channel_tx.reshape(-1)
        L = channel_in.shape[0] 
        channel_in = channel_in[:L // 2] + channel_in[L // 2:] * 1j # chuyển thành số phức mỗi số có chiều /2 
       
        """L/2 giá trị đầu tiên khi reshape sẽ thành phần thực, L/2 giá trị sau là phần ảo và chúng tương ứng với nhau
        và được ghép lại thành số phức là channel_in"""
        channel_output = self.complex_forward(channel_in, snr)
        channel_output = torch.cat([torch.real(channel_output), torch.imag(channel_output)]) 

        # torch real sẽ tách phần thực thành 1 tensor 1D và torch imag sẽ tách phần ảo thành 1 tensor 1D rồi ghép ngay sau phần thực
        channel_output = channel_output.reshape(input_shape) # trở về B, num_channel, out_dim
        #print("channel_output.shape: ", channel_output.shape) # B, num_channel, out_dim
        if self.channel_type == 'AWGN':
            noise = (channel_output - channel_tx).detach() # tách noise 
            noise.requires_grad = False # không cho phép gradient của noise được tính toán trong quá trình backpropagation để encoder k học từ noise 
            channel_tx = channel_tx + noise
            if avg_pwr:
                return channel_tx * torch.sqrt(avg_pwr * 2)
            else:
                return channel_tx * torch.sqrt(pwr) # có cần chuẩn hóa công suất hay không 
                #print("channel_tx.shape: ", channel_tx * torch.sqrt(pwr)) # B, num_channel, out_dim
        elif self.channel_type == 'Rayleigh':
            if avg_pwr:
                return channel_output * torch.sqrt(avg_pwr * 2)
            else:
                return channel_output * torch.sqrt(pwr)
        #print("channel_output.shape: ", channel_output.shape) # B, num_channel, out_dim
    def complex_forward(self, channel_in, chan_param):
        # if self.chan_type == 0 or self.chan_type == 'none': # noiseless thì return lại channel in lúc này là các số phức 
        #     return channel_in
        if self.channel_type == "None":
            print("No channel")
            return channel_in
        elif self.channel_type == 'AWGN':
           
            channel_tx = channel_in
            sigma = np.sqrt(1.0 / (2 * 10 ** (chan_param / 10))) # từ chan_param tính ra sigma noise std 
            chan_output = self.gaussian_noise_layer(channel_tx,
                                                    std=sigma,
                                                    name="awgn_chan_noise")
            
            return chan_output # channel_output lúc này vẫn là số phưcs đã được cộng thêm noise 

        elif self.channel_type == 'Rayleigh':
            channel_tx = channel_in
            sigma = np.sqrt(1.0 / (2 * 10 ** (chan_param / 10)))
            chan_output = self.rayleigh_noise_layer(channel_tx,
                                                    std=sigma,
                                                    name="rayleigh_chan_noise")
            
            return chan_output


    def noiseless_forward(self, channel_in):
        channel_tx = self.normalize(channel_in, power=1)
        return channel_tx

