import torch
import torch.nn as nn
from channels.channel_base import Channel


class BaseModel(nn.Module):
    def __init__(self, args, in_channel, class_num):
        super(BaseModel, self).__init__()

        self.in_channel = in_channel
        self.class_num = class_num
        self.inv_cdim = args.inv_cdim  # int(32)  # inv_cdim
        self.var_cdim = args.var_cdim  # int(32)  # var_cdim
        self.ib_cdim = self.inv_cdim + self.var_cdim
        self.P = 1
        self.e = 1e-24
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu", index=0)
