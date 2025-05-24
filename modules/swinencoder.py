from modules.swinjscc_module import * 
import torch

class SwinTransformerBlock(nn.Module):

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,

                 mlp_ratio=4., qkv_bias=True, qk_scale=None, act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution 
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size: # kích thước nhỏ nhất đầu vào <= cửa sổ thì là W-MSA(shift_size = 0)
            # if window size is larger than input resolution, we don't partition windows 
            self.shift_size = 0 # k dùng shifted và window size chính bằng cái min
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

# FIG 6a 
        self.norm1 = norm_layer(dim) # LN 
        self.attn = WindowAttention( # Shifted window for MSA 
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale)

        self.norm2 = norm_layer(dim) # LN trước khi vaof MLP 
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):

        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        # Swin transformer block bên trái 
        shortcut = x
        x = self.norm1(x) # LN 
        x = x.view(B, H, W, C) # Đổi lại kích thước về B, H, W, C 

        # cyclic shift
        if self.shift_size > 0: # SW-MSA
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x # W-MSA

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C
        B_, N, C = x_windows.shape

        # merge windows
        attn_windows = self.attn(x_windows,
                                 add_token=False,
                                 mask=self.attn_mask)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))

        return x

    # def extra_repr(self) -> str:
    #     return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
    #            f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    # def flops(self):
    #     flops = 0
    #     H, W = self.input_resolution
    #     # norm1
    #     flops += self.dim * H * W
    #     # W-MSA/SW-MSA
    #     nW = H * W / self.window_size / self.window_size
    #     flops += nW * self.attn.flops(self.window_size * self.window_size)
    #     # mlp
    #     flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
    #     # norm2
    #     flops += self.dim * H * W
    #     return flops

    def update_mask(self):
        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
            self.attn_mask = attn_mask.cuda()
        else:
            pass


class BasicLayer(nn.Module):
    def __init__(self, dim, out_dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, norm_layer=nn.LayerNorm,
                 downsample=None):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=out_dim,
                                 input_resolution=(input_resolution[0] // 2, input_resolution[1] // 2),
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, out_dim=out_dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        if self.downsample is not None:
            x = self.downsample(x) # downsample = patch merging layer 
        for _, blk in enumerate(self.blocks):
            x = blk(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops

    def update_resolution(self, H, W):
        for _, blk in enumerate(self.blocks):
            blk.input_resolution = (H, W)
            blk.update_mask()
        if self.downsample is not None:
            self.downsample.input_resolution = (H * 2, W * 2)

class AdaptiveModulator(nn.Module):  # SM module
    def __init__(self, M):
        super(AdaptiveModulator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(1, M),
            nn.ReLU(),
            nn.Linear(M, M),
            nn.ReLU(),
            nn.Linear(M, M),
            nn.Sigmoid()
        )

    def forward(self, snr):
        # Đảm bảo snr nằm trên cùng thiết bị với self.fc
        snr = snr.to(next(self.fc.parameters()).device)
        return self.fc(snr)

class SwinJSCC_Encoder(nn.Module):
    def __init__(self, img_size, patch_size, in_chans,
                 embed_dims, depths, num_heads, C,
                 window_size=4, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 norm_layer=nn.LayerNorm, patch_norm=True,
                 bottleneck_dim=16, device='cpu'):  # Thêm tham số device
        super().__init__()
        self.device = device  # Lưu thiết bị vào self.device
        self.num_layers = len(depths) # num_stage 
        self.num_features = bottleneck_dim
        self.mlp_ratio = mlp_ratio
        self.embed_dims = embed_dims
        self.in_chans = in_chans
        self.patch_size = patch_size
        self.patches_resolution = img_size
        
        self.H = img_size[0] // (2 ** self.num_layers)
        self.W = img_size[1] // (2 ** self.num_layers)
        self.patch_embed = PatchEmbed(img_size, 2, 3, embed_dims[0]) # ìmg_size, patch_size(2x2), in_chans(img), embed_dim
        #=> tensor B, num_patches(H/2*W/2), embed_dim(C1) 
        self.hidden_dim = int(self.embed_dims[len(embed_dims)-1] * 1.5)
        self.layer_num = layer_num = 7 # 7 tầng SM 

        # build layers
        self.layers = nn.ModuleList() 
        for i_layer in range(self.num_layers): # num_layers = 4 là stage 
            layer = BasicLayer(dim=int(embed_dims[i_layer - 1]) if i_layer != 0 else 3,
                               out_dim=int(embed_dims[i_layer]),
                               input_resolution=(self.patches_resolution[0] // (2 ** i_layer), # đầu vào mỗi stage thì kích thước đã bị
                                                 self.patches_resolution[1] // (2 ** i_layer)), # chia nhỏ tương ứng 
                               depth=depths[i_layer], 
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               norm_layer=norm_layer,
                               downsample=PatchMerging if i_layer != 0 else None)

            self.layers.append(layer) # thêm các stage  vào list
        self.norm = norm_layer(embed_dims[-1]) # LayerNorm. số chiều stage cuối 
        if C != None: # nếu không dùng Rate Modnet 
            self.head_list = nn.Linear(embed_dims[-1], C)  
        self.apply(self._init_weights)
 ### Channel ModNet ####### 
        self.bm_list = nn.ModuleList() # chứa SM module thứ i biến đổi SNR 
        self.sm_list = nn.ModuleList() # dùng cho tạo các khối FC ngay sau SM còn BM chứa SM tạo các khối FC bên dưới 
        self.sm_list.append(nn.Linear(self.embed_dims[len(embed_dims) - 1], self.hidden_dim)) # thêm cái FC Ci -> N đầu tiên
        for i in range(layer_num): # 0-> 6
            if i == layer_num - 1: # nếu đã đến layer cuối thì số channel trở về Ci ban đầu 
                outdim = self.embed_dims[len(embed_dims) - 1] 
            else:
                outdim = self.hidden_dim # nếu chưa thì vẫn bằng hidden_num(N)
            self.bm_list.append(AdaptiveModulator(self.hidden_dim))
            self.sm_list.append(nn.Linear(self.hidden_dim, outdim))
        self.sigmoid = nn.Sigmoid()

        self.bm_list1 = nn.ModuleList()
        self.sm_list1 = nn.ModuleList()
        self.sm_list1.append(nn.Linear(self.embed_dims[len(embed_dims) - 1], self.hidden_dim))
        # từ y' số channel là C4(embed_dims[-1]) -> hidden_dim (*1.5)
        for i in range(layer_num): # 0->7 
            if i == layer_num - 1: # đến 6 là tầng cuối rồi 
                outdim = self.embed_dims[len(embed_dims) - 1]
            else:
                outdim = self.hidden_dim
            self.bm_list1.append(AdaptiveModulator(self.hidden_dim)) # Chính là SM module với M(orN) = hidden_dim
            self.sm_list1.append(nn.Linear(self.hidden_dim, outdim)) # là Linear layer(FC) với đầu vào là hidden_dim, ra là chính nó hoặc Ci
        self.sigmoid1 = nn.Sigmoid() # Cuói cùng qua sigmoid 

    def forward(self, x, snr, rate, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):  # Thêm tham số device với giá trị mặc định là 'cpu'
        B, C, H, W = x.size() 

        x = self.patch_embed(x) # Giảm kích thước ảnh xuống còn H/2, W/2, số channel là C1
        for i_layer, layer in enumerate(self.layers): # đang ở stage mấy 
            x = layer(x) # x đi qua từng tầng stage 
        x = self.norm(x)

        snr_cuda = torch.tensor(snr, dtype=torch.float).to(device)  # Sử dụng device được truyền vào
        rate_cuda = torch.tensor(rate, dtype=torch.float).to(device)
        snr_batch = snr_cuda.unsqueeze(0).expand(B, -1)
        rate_batch = rate_cuda.unsqueeze(0).expand(B, -1)
        for i in range(self.layer_num):
            if i == 0:
                temp = self.sm_list1[i](x.detach())
            else:
                temp = self.sm_list1[i](temp)

            bm = self.bm_list1[i](snr_batch).unsqueeze(1).expand(-1, H * W // (self.num_layers ** 4), -1)
            temp = temp * bm
        mod_val1 = self.sigmoid1(self.sm_list1[-1](temp))
        x = x * mod_val1

        for i in range(self.layer_num):
            if i == 0:
                temp = self.sm_list[i](x.detach())
            else:
                temp = self.sm_list[i](temp)

            bm = self.bm_list[i](rate_batch).unsqueeze(1).expand(-1, H * W // (self.num_layers ** 4), -1)
            temp = temp * bm
        mod_val = self.sigmoid(self.sm_list[-1](temp)) # đầu ra layer RM cuối cùng sẽ đi qua sigmoid

        x = x * mod_val # B, num_patches, C4(2)

        mask = torch.sum(mod_val, dim=1)
        sorted, indices = mask.sort(dim=1, descending=True)

        c_indices = indices[:, :rate] # Giữ lại rate kênh lớn nhất(số kênh vào channel)

        # add = torch.Tensor(range(0, B * x.size()[2], x.size()[2])).unsqueeze(1).repeat(1, rate)
        # c_indices = c_indices + add.int().cuda()
        """Chuyển về cùng thiết bị"""
        add = torch.Tensor(range(0, B * x.size()[2], x.size()[2])).unsqueeze(1).repeat(1, rate).to(device) 
        # Tạo tensor: [0, C(x), 2C(x), ..., B*C(x)] C(x) là số kênh đầu ra encoder(C2 or C4)

        c_indices = c_indices + add.int()
        mask = torch.zeros(mask.size()).reshape(-1).cuda()
        mask[c_indices.reshape(-1)] = 1
        mask = mask.reshape(B, x.size()[2])
        mask = mask.unsqueeze(1).expand(-1, H * W // (self.num_layers ** 4), -1)

        mask = mask.to(x.device)  # Chuyển mask về cùng thiết bị với x
        x = x * mask
        return x, mask



        # if model == 'SwinJSCC_w/o_SAandRA':
        #     x = self.head_list(x)
        #     return x

        # elif model == 'SwinJSCC_w/_SA':
        #     snr_cuda = torch.tensor(snr, dtype=torch.float).to(device) # đưa SNR thành tensor gửi lên device 
        #     snr_batch = snr_cuda.unsqueeze(0).expand(B, -1)  # mở rộng thành (B, snr), tất cả batch có cùng SNR
        #     for i in range(self.layer_num): # layer_num là stt của SM module 
        #         if i == 0:
        #             temp = self.sm_list[i](x.detach()) # x.detach() khong tính toán gradient 
        #         else:
        #             temp = self.sm_list[i](temp) # đi qua từng lớp FC, SM 

        #         bm = self.bm_list[i](snr_batch).unsqueeze(1).expand(-1, H * W // (self.num_layers ** 4), -1)
        #         # Thêm một chiều vào vị trí 1 -> B,1,hidden_dim và vị trí 1 kia thay thành H*W/(num_layers^4)
        #         # nếu qua cả 4 stage rồi thì là H*W/16*16 hay H/16 * W/16 hợp lý 
        #         temp = temp * bm
        #     mod_val = self.sigmoid(self.sm_list[-1](temp))#cuối cùng trong sm_list là FC|N x Ci 
        #     x = x * mod_val # 
        #     x = self.head_list(x)
        #     return x
        # #RA: các khối FC bên ngoài thì vẫn vậy chỉ khác khối SM trong bm thì thay thành RM vẫn tương tự cấu trúc
        # # chỉ khác đầu vào là R không phải SNR
        # elif model == 'SwinJSCC_w/_RA':
        #     rate_cuda = torch.tensor(rate, dtype=torch.float).to(device)
        #     rate_batch = rate_cuda.unsqueeze(0).expand(B, -1)
        #     for i in range(self.layer_num):
        #         if i == 0:
        #             temp = self.sm_list[i](x.detach())
        #         else:
        #             temp = self.sm_list[i](temp) 

        #         bm = self.bm_list[i](rate_batch).unsqueeze(1).expand(-1, H * W // (self.num_layers ** 4), -1)
        #         temp = temp * bm # temp(B, num_patches(H*W//num_layer ** 4), out_dim)
        #     mod_val = self.sigmoid(self.sm_list[-1](temp))
        #     x = x * mod_val
        #     mask = torch.sum(mod_val, dim=1) # tính tổng theo num_patches -> mask (B,C)
        #     sorted, indices = mask.sort(dim=1, descending=True) 
        #     # sắp xếp thứ tự giảm dần độ quan trọng của chiều channel để chọn xem mask channel nào  
        #     c_indices = indices[:, :rate] 
        #     # lấy vị trí rate(C4/C_in) kênh quan trọng nhất với C4 lấy từ embed_dims 
        #     add = torch.Tensor(range(0, B * x.size()[2], x.size()[2])).unsqueeze(1).repeat(1, rate)
        #     # x đã qua Layer rồi nên thành tensỏr 3 chiều (B, num_patches, out_dim)
        #     # Chuyển thành tensor từ range [0, C , 2C,... BxC] với kích thước là (B, rate)
        #     c_indices = c_indices + add.int().cuda() 
        #     mask = torch.zeros(mask.size()).reshape(-1).cuda() # mask toàn 0, một chiều(reshape(-1) tự tính toán chiều còn lại)
        #     mask[c_indices.reshape(-1)] = 1 # Những kênh được chọn thì thành 1
        #     mask = mask.reshape(B, x.size()[2]) # Chuyển lại về BxC (out_dim)
        #     mask = mask.unsqueeze(1).expand(-1, H * W // (self.num_layers ** 4), -1) # B, num_patches, out_dim
        #     x = x * mask
        #     return x, mask

        # elif model == 'SwinJSCC_w/_SAandRA':
        #     snr_cuda = torch.tensor(snr, dtype=torch.float).to(device)
        #     rate_cuda = torch.tensor(rate, dtype=torch.float).to(device)
        #     snr_batch = snr_cuda.unsqueeze(0).expand(B, -1)
        #     rate_batch = rate_cuda.unsqueeze(0).expand(B, -1)
        #     for i in range(self.layer_num):
        #         if i == 0:
        #             temp = self.sm_list1[i](x.detach())
        #         else:
        #             temp = self.sm_list1[i](temp)

        #         bm = self.bm_list1[i](snr_batch).unsqueeze(1).expand(-1, H * W // (self.num_layers ** 4), -1)
        #         temp = temp * bm
        #     mod_val1 = self.sigmoid1(self.sm_list1[-1](temp))
        #     x = x * mod_val1

        #     for i in range(self.layer_num):
        #         if i == 0:
        #             temp = self.sm_list[i](x.detach())
        #         else:
        #             temp = self.sm_list[i](temp)

        #         bm = self.bm_list[i](rate_batch).unsqueeze(1).expand(-1, H * W // (self.num_layers ** 4), -1)
        #         temp = temp * bm
        #     mod_val = self.sigmoid(self.sm_list[-1](temp)) #
        #     x = x * mod_val
        #     mask = torch.sum(mod_val, dim=1)
        #     sorted, indices = mask.sort(dim=1, descending=True)
        #     c_indices = indices[:, :rate]
        #     add = torch.Tensor(range(0, B * x.size()[2], x.size()[2])).unsqueeze(1).repeat(1, rate)
        #     c_indices = c_indices + add.int().cuda()
        #     mask = torch.zeros(mask.size()).reshape(-1).cuda()
        #     mask[c_indices.reshape(-1)] = 1
        #     mask = mask.reshape(B, x.size()[2])
        #     mask = mask.unsqueeze(1).expand(-1, H * W // (self.num_layers ** 4), -1)

        #     x = x * mask
        #     return x, mask

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # @torch.jit.ignore
    # def no_weight_decay(self):
    #     return {'absolute_pos_embed'}

    # @torch.jit.ignore
    # def no_weight_decay_keywords(self):
    #     return {'relative_position_bias_table'}
    # # tính toán lượng phép tính (độ phức tạp)
    # def flops(self):
    #     flops = 0
    #     flops += self.patch_embed.flops()
    #     for i, layer in enumerate(self.layers):
    #         flops += layer.flops()
    #     flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
    #     return flops

    def update_resolution(self, H, W):
        self.input_resolution = (H, W)
        for i_layer, layer in enumerate(self.layers):
            layer.update_resolution(H // (2 ** (i_layer + 1)),
                                    W // (2 ** (i_layer + 1)))


def create_encoder(**kwargs):
    model = SwinJSCC_Encoder(**kwargs)
    return model


