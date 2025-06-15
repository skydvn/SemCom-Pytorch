# import torch
# import torch.nn as nn
# import numpy as np
# import torch.nn.functional as F
# from random import choice
# from channels.channel_base import Channel
# from models.model_base import BaseModel
# from modules.swinencoder import create_encoder
# from modules.swindecoder import create_decoder
# from collections import OrderedDict
# # try:
# #     from backpack import backpack, extend
# #     from backpack.extensions import BatchGrad
# # except:
# #     backpack = None

# # from config import config
# class SWINJSCC(BaseModel):
#     def __init__(self, args, in_channel, class_num):
#         super(SWINJSCC, self).__init__(args, in_channel, class_num)
#         self.args = args
#         # print("sdfdfs", args.ratio)
#         # print("egwreg",args.base_snr)
#         if isinstance(args.ratio, list):
#             raise ValueError(f"args.ratio must be a single value, not a list: {args.ratio}")
#         self.squared_difference = torch.nn.MSELoss(reduction='none')
#         # self.distortion_loss = Distortion(args)
#         self.in_channel = in_channel
#         self.class_num = class_num
#         self.downsample = args.downsample
#         encoder_kwargs = args.encoder_kwargs
#         decoder_kwargs = args.decoder_kwargs
#         self.encoder = create_encoder(**encoder_kwargs)
#         self.decoder = create_decoder(**decoder_kwargs)
#         self.device = torch.device("cuda" if torch.cuda.is_available() and args.device else "cpu")
#         self.pass_channel = args.pass_channel
#         self.H = self.W = 0
#         self.name = "SwinJSCC"
#         #self.multiple_snr = [int(snr) for snr in args.snr_list]
#         #self.snr = int(args.base_snr)
#         #self.channel = Channel(channel_type="AWGN", snr=self.snr)
#         self.channel_number = int(args.ratio * (2 * 3 * 2 ** (self.downsample * 2)))
#     def feature_pass_channel(self, feature):
#         noisy_feature = self.channel(feature)  # Loại bỏ avg_pwr
#         return noisy_feature


#     def forward(self, input_image,snr_chan):  # input_image: là x
#         B, _, H, W = input_image.shape
#         #print("Channel swin is", self.channel.get_channel())

#         if H != self.H or W != self.W:
#             self.encoder.update_resolution(H, W)
#             self.decoder.update_resolution(H // (2 ** self.downsample), W // (2 ** self.downsample))
#             self.H = H
#             self.W = W
#         feature, mask = self.encoder(input_image, snr_chan, self.channel_number)

#         CBR = self.channel_number / (2 * 3 * 2 ** (self.downsample * 2))
#         avg_pwr = torch.sum(feature ** 2) / mask.sum()

#         if self.pass_channel:
#             B, L, C = feature.shape
#             H_patch = input_image.shape[2] // (2**self.downsample)
#             W_patch = input_image.shape[3] // (2**self.downsample)
#             assert H_patch * W_patch == L, (
#             f"Mismatch tokens: L={L} nhưng H_patch×W_patch="
#             f"{H_patch}×{W_patch}={H_patch*W_patch}"
#             )
#             feature_4D = feature.reshape(B, H_patch, W_patch, C).permute(0, 3, 1, 2)  # Chuyển đổi về (B, C, H, W)

#             # Qua kênh
#             noisy_feature_4D = self.feature_pass_channel(feature_4D)

#             # Chuyển đổi noisy_feature về 3D để truyền vào decoder
#             noisy_feature = noisy_feature_4D.flatten(2).permute(0, 2, 1)  # Chuyển đổi về (B, L, C)
#         else:
#             noisy_feature = feature

#         noisy_feature = noisy_feature * mask
#         # Decode
#         recon_image = self.decoder(noisy_feature, snr_chan)

#         return recon_image, CBR, snr_chan

#     def get_latent(self, x):
#         enc,_ = self.encoder(x)
#         enc = self.normalize_layer(enc)
#         return enc

#     def get_train_recon(self, x, base_snr):
#         z = self.encoder(x)
#         z = self.normalize_layer(z)
#         z = self.channel(z)

#         x_hat = self.decoder(z)
#         return x_hat

#     def get_latent_size(self, x):
#         enc = self.encoder(x)
#         enc = self.normalize_layer(enc)
#         return enc.size()

#     def normalize_layer(self, z):
#         k = torch.tensor(1.0).to(self.device)  # torch.prod(torch.tensor(z.size()[1:], dtype=torch.float32))
#         # Square root of k and P
#         sqrt1 = torch.sqrt(k * self.P)
#         sqrt2 = torch.sqrt(z*z + self.e)
#         div = z / sqrt2
#         z_out = div * sqrt1

#         return z_out
#     def change_channel(self, channel_type='AWGN', snr=None):
#         if snr is None:
#             self.channel = None
#         else:
#             self.channel = Channel(channel_type, snr)

#     def get_channel(self):
#         if hasattr(self, 'channel') and self.channel is not None:
#             return self.channel.get_channel()
#         return None
    
#     def channel_perturb(self, input_image, chan_type, snr_chan):
#         B, _, H, W = input_image.shape

#         if H != self.H or W != self.W:
#             self.encoder.update_resolution(H, W)
#             self.decoder.update_resolution(H // (2 ** self.downsample), W // (2 ** self.downsample))
#             self.H = H
#             self.W = W
#         feature, mask = self.encoder(input_image, snr_chan, self.channel_number)
#         #print("Featureeee", feature)
#         # CBR = self.channel_number / (2 * 3 * 2 ** (self.downsample * 2))
#         # avg_pwr = torch.sum(feature ** 2) / mask.sum()

#         if self.pass_channel:
#             # Chuyển đổi feature về 4D trước khi qua kênh
#             #print("Name of channellll: ", self.channel.get_channel()) 
#             B, L, C = feature.shape
#             H_patch = input_image.shape[2] // (2**self.downsample)
#             W_patch = input_image.shape[3] // (2**self.downsample)
#             assert H_patch * W_patch == L, (
#             f"Mismatch tokens: L={L} nhưng H_patch×W_patch="
#             f"{H_patch}×{W_patch}={H_patch*W_patch}"
#             )
#             #H = W = int(L**0.5)  # Giả định L là số lượng patch (H * W)
#             feature_4D = feature.reshape(B, H_patch, W_patch, C).permute(0, 3, 1, 2)  # Chuyển đổi về (B, C, H, W)

#             # Qua kênh
#             self.change_channel(channel_type=chan_type, snr=snr_chan)
#             #print("Name of channel: ", self.channel.get_channel())
#             noisy_feature_4D = self.feature_pass_channel(feature_4D)

#             # Chuyển đổi noisy_feature về 3D để truyền vào decoder
#             noisy_feature = noisy_feature_4D.flatten(2).permute(0, 2, 1)  # Chuyển đổi về (B, L, C)
#         else:
#             noisy_feature = feature

#         noisy_feature = noisy_feature * mask
#         # Decode
#         recon_image = self.decoder(noisy_feature, snr_chan)

#         return recon_image



import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from random import choice
from channels.channel_base import Channel
from models.model_base import BaseModel
from modules.swinencoder import create_encoder
from modules.swindecoder import create_decoder
# from config import config
class SWINJSCC_FISHR(BaseModel):
    def __init__(self, args, in_channel, class_num):
        super(SWINJSCC_FISHR, self).__init__(args, in_channel, class_num)
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
        #self.multiple_snr = [int(snr) for snr in args.snr_list]
        
        #self.channel = Channel(channel_type="AWGN", snr=self.snr)
        self.channel_number = int(args.ratio * (2 * 3 * 2 ** (self.downsample * 2)))
    def feature_pass_channel(self, feature):
        noisy_feature = self.channel.forward(feature) 
        return noisy_feature


    def forward(self, input_image,snr_chan):  # input_image: là x
        B, _, H, W = input_image.shape
        #print("Channel swin is", self.channel.get_channel())

        if H != self.H or W != self.W:
            self.encoder.update_resolution(H, W)
            self.decoder.update_resolution(H // (2 ** self.downsample), W // (2 ** self.downsample))
            self.H = H
            self.W = W
        feature, mask = self.encoder(input_image, snr_chan, self.channel_number)

        CBR = self.channel_number / (2 * 3 * 2 ** (self.downsample * 2))
        avg_pwr = torch.sum(feature ** 2) / mask.sum()

        if self.pass_channel:
            B, L, C = feature.shape
            H_patch = input_image.shape[2] // (2**self.downsample)
            W_patch = input_image.shape[3] // (2**self.downsample)
            assert H_patch * W_patch == L, (
            f"Mismatch tokens: L={L} nhưng H_patch×W_patch="
            f"{H_patch}×{W_patch}={H_patch*W_patch}"
            )
            feature_4D = feature.reshape(B, H_patch, W_patch, C).permute(0, 3, 1, 2)  # Chuyển đổi về (B, C, H, W)

            # Qua kênh
            noisy_feature_4D = self.feature_pass_channel(feature_4D)

            # Chuyển đổi noisy_feature về 3D để truyền vào decoder
            noisy_feature = noisy_feature_4D.flatten(2).permute(0, 2, 1)  # Chuyển đổi về (B, L, C)
        else:
            noisy_feature = feature

        noisy_feature = noisy_feature * mask
        # Decode
        recon_image = self.decoder(noisy_feature, snr_chan)

        return recon_image

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
    def channel_perturb(self, input_image, chan_type, snr_chan):
        B, _, H, W = input_image.shape

        if H != self.H or W != self.W:
            self.encoder.update_resolution(H, W)
            self.decoder.update_resolution(H // (2 ** self.downsample), W // (2 ** self.downsample))
            self.H = H
            self.W = W
        feature, mask = self.encoder(input_image, snr_chan, self.channel_number)

        B, L, C = feature.shape
        H_patch = input_image.shape[2] // (2**self.downsample)
        W_patch = input_image.shape[3] // (2**self.downsample)
        assert H_patch * W_patch == L, (
        f"Mismatch tokens: L={L} nhưng H_patch×W_patch="
        f"{H_patch}×{W_patch}={H_patch*W_patch}"
            )
            #H = W = int(L**0.5)  # Giả định L là số lượng patch (H * W)
        feature_4D = feature.reshape(B, H_patch, W_patch, C).permute(0, 3, 1, 2)  # Chuyển đổi về (B, C, H, W)

            # Qua kênh
        self.change_channel(channel_type=chan_type, snr=snr_chan)
            #print("Name of channel: ", self.channel.get_channel())
        noisy_feature_4D = self.feature_pass_channel(feature_4D)

            # Chuyển đổi noisy_feature về 3D để truyền vào decoder
        noisy_feature = noisy_feature_4D.flatten(2).permute(0, 2, 1)  # Chuyển đổi về (B, L, C)


        noisy_feature = noisy_feature * mask
        # Decode
        recon_image = self.decoder(noisy_feature, snr_chan)

        return recon_image