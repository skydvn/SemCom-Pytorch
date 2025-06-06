import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from random import choice
from channels.channel import Channel
from models.model_base import BaseModel
from modules.swinencoder import create_encoder
from modules.swindecoder import create_decoder
# from config import config
class NEWSWINJSCC(BaseModel):
    def __init__(self, args, in_channel, class_num):
        super(NEWSWINJSCC, self).__init__(args, in_channel, class_num)
        self.args = args
        print("sdfdfs", args.ratio)
        print("egwreg",args.base_snr)
        if isinstance(args.ratio, list):
            raise ValueError(f"args.ratio must be a single value, not a list: {args.ratio}")
        self.squared_difference = torch.nn.MSELoss(reduction='none')
        # self.distortion_loss = Distortion(args)
        self.in_channel = in_channel
        self.class_num = class_num
        self.downsample = args.downsample
        encoder_kwargs = args.encoder_kwargs
        decoder_kwargs = args.decoder_kwargs
        self.encoder = create_encoder(**encoder_kwargs)
        self.decoder = create_decoder(**decoder_kwargs)
        self.device = torch.device("cuda" if torch.cuda.is_available() and args.device else "cpu")
        self.pass_channel = args.pass_channel
        self.H = self.W = 0
        self.name = "SwinJSCC"
        #self.multiple_snr = [int(snr) for snr in args.snr_list]
        self.snr = int(args.base_snr)
        self.channel = Channel(channel_type="AWGN", snr=self.snr)
        self.channel_number = int(args.ratio * (2 * 3 * 2 ** (self.downsample * 2)))
    def feature_pass_channel(self, feature, chan_param,avg_pwr=None):
        noisy_feature = self.channel.forward(feature, chan_param,avg_pwr)  # Loại bỏ avg_pwr
        return noisy_feature


    def forward(self, input_image, given_SNR=None, given_rate=None):  # input_image: là x
        B, _, H, W = input_image.shape
        print()
        if H != self.H or W != self.W:
            self.encoder.update_resolution(H, W)
            self.decoder.update_resolution(H // (2 ** self.downsample), W // (2 ** self.downsample))
            self.H = H
            self.W = W
        feature, mask = self.encoder(input_image, self.snr, self.channel_number)

        CBR = self.channel_number / (2 * 3 * 2 ** (self.downsample * 2))
        avg_pwr = torch.sum(feature ** 2) / mask.sum()
        noisy_feature = self.feature_pass_channel(feature,self.snr,avg_pwr)
        noisy_feature = noisy_feature * mask
        # Decode
        recon_image = self.decoder(noisy_feature, self.snr)

        return recon_image, CBR, self.snr

    def get_latent(self, x):
        enc,_ = self.encoder(x)
        enc = self.normalize_layer(enc)
        return enc

    def get_train_recon(self, x, base_snr):
        z = self.encoder(x)
        z = self.normalize_layer(z)
        z = self.channel(z)

        x_hat = self.decoder(z)
        return x_hat

    def get_latent_size(self, x):
        enc = self.encoder(x)
        enc = self.normalize_layer(enc)
        return enc.size()

    def normalize_layer(self, z):
        k = torch.tensor(1.0).to(self.device)  # torch.prod(torch.tensor(z.size()[1:], dtype=torch.float32))
        # Square root of k and P
        sqrt1 = torch.sqrt(k * self.P)

        # Conjugate Transpose of z
        # if torch.is_complex(z):
        #     zT = torch.conj(z).permute(0, 1, 3, 2)
        # else:
        #     zT = z.permute(0, 1, 3, 2)
        # Multiply z and zT = sqrt2
        sqrt2 = torch.sqrt(z*z + self.e)
        div = z / sqrt2
        z_out = div * sqrt1

        return z_out

    def change_channel(self, channel_type='AWGN', snr=None):
        if snr is None:
            self.channel = None
        else:
            self.channel = Channel(channel_type, snr)

    def get_channel(self):
        if hasattr(self, 'channel') and self.channel is not None:
            return self.channel.get_channel()
        return None
    def channel_perturb(self, input_image, domain_list):
        B, _, H, W = input_image.shape

        if H != self.H or W != self.W:
            self.encoder.update_resolution(H, W)
            self.decoder.update_resolution(H // (2 ** self.downsample), W // (2 ** self.downsample))
            self.H = H
            self.W = W
        feature, mask = self.encoder(input_image, self.snr, self.channel_number)

        self.change_channel(channel_type=domain_list, snr = self.snr)
        avg_pwr = torch.sum(feature ** 2) / mask.sum()
        noisy_feature = self.feature_pass_channel(feature,self.snr,avg_pwr)
        noisy_feature = noisy_feature * mask
        # Decode
        recon_image = self.decoder(noisy_feature, self.snr)

        return recon_image
