# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# MAE: https://github.com/facebookresearch/mae
# --------------------------------------------------------
import math
from functools import partial

import torch
import torch.nn as nn
from timm.models.vision_transformer import PatchEmbed, Mlp

from util.pos_embed import get_2d_sincos_pos_embed
from util.blocks import GaussianConv2d
from util.blocks import Block_SelfMask, Block_SelfCrossMask


class AimViT(nn.Module):
    """
    Pretrain vision transformer backbone with AIM
    parall encoder-decoder architecture
    Modified by sky: use the blocks in ViT (+ mask) for encoders, which is more convinent for finetune, linear
    modify the permutation form stochastic mask to center-out mask
    """

    def __init__(self,
                 # vision transformer backbone
                 img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16, drop_path_rate=0., out_dim=768,
                 mlp_ratio=4., norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 # aim
                 permutation_type='center2out', attention_type='cls',
                 # decoder
                 query_depth=12, share_weight=False,
                 prediction_head_type='MLP',
                 # loss function
                 gaussian_kernel_size=None, gaussian_sigma=None,
                 loss_type='L2', predict_feature='none', norm_pix_loss=True):
        super().__init__()

        # patch embedding
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.patch_size = patch_size

        # cls token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # position embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        # encoder
        self.blocks = nn.ModuleList([
            Block_SelfMask(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer, drop_path=dpr[i])
            for i in range(depth)])

        # decoder
        if share_weight:
            self.query_blocks = self.blocks
        else:
            self.query_blocks = nn.ModuleList([
                Block_SelfCrossMask(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer, drop_path=dpr[i])
                for i in range(query_depth)])
        self.depth = depth
        self.step = depth // query_depth

        # permutation type
        self.permutation_type = permutation_type

        # prediction head
        self.norm = norm_layer(embed_dim)
        self.predict_feature = predict_feature
        self.attention_type = attention_type
        if prediction_head_type == 'LINEAR':
            if predict_feature == 'none':
                self.prediction_head = nn.Linear(embed_dim, patch_size ** 2 * 3)
            else:
                rec_dim = out_dim if predict_feature == 'clip' else embed_dim
                self.prediction_head = nn.Linear(embed_dim, rec_dim)
        elif prediction_head_type == 'MLP':
            if predict_feature == 'none':
                self.prediction_head = Mlp(embed_dim, int(embed_dim * mlp_ratio), patch_size ** 2 * 3)
            else:
                rec_dim = out_dim if predict_feature == 'clip' else embed_dim
                self.prediction_head = Mlp(embed_dim, int(embed_dim * mlp_ratio), rec_dim)

        # define loss parameters
        self.loss_type = loss_type
        self.norm_pix_loss = norm_pix_loss
        if gaussian_kernel_size is not None and gaussian_sigma is not None and self.predict_feature == 'none':
            self.gaussian_blur = GaussianConv2d(3, gaussian_kernel_size, gaussian_sigma)
        else:
            self.gaussian_blur = nn.Identity()

        # spilit matrix for guided center permutation
        num_patch = img_size // patch_size
        split_matrix = torch.zeros((num_patch, 2, 4))
        split_matrix[0, :, :] = torch.tensor([[0, 0, 0, 0], [2, 6, 10, 13]])
        split_matrix[1, :, :] = torch.tensor([[0, 0, 0, 0], [2, 6, 10, 13]])
        split_matrix[2, :, :] = torch.tensor([[0, 0, 0, 0], [2, 6, 10, 13]])
        split_matrix[3, :, :] = torch.tensor([[2, 0, 0, 0], [4, 6, 10, 13]])
        split_matrix[4, :, :] = torch.tensor([[3, 1, 0, 0], [5, 7, 10, 13]])
        split_matrix[5, :, :] = torch.tensor([[4, 2, 0, 0], [6, 8, 10, 13]])
        split_matrix[6, :, :] = torch.tensor([[5, 3, 1, 0], [7, 9, 11, 13]])
        split_matrix[7, :, :] = torch.tensor([[6, 4, 2, 0], [8, 10, 12, 13]])
        split_matrix[8, :, :] = torch.tensor([[7, 5, 3, 0], [9, 11, 13, 13]])
        split_matrix[9, :, :] = torch.tensor([[8, 6, 3, 0], [10, 12, 13, 13]])
        split_matrix[10, :, :] = torch.tensor([[9, 7, 3, 0], [11, 13, 13, 13]])
        split_matrix[11, :, :] = torch.tensor([[11, 7, 3, 0], [13, 13, 13, 13]])
        split_matrix[12, :, :] = torch.tensor([[11, 7, 3, 0], [13, 13, 13, 13]])
        split_matrix[13, :, :] = torch.tensor([[11, 7, 3, 0], [13, 13, 13, 13]])
        self.split_matrix = split_matrix

        # coordinates for patches (row, col)
        coordinates = torch.zeros((num_patches, 2))
        for i in range(num_patch):
            for j in range(num_patch):
                coordinates[i*num_patch+j, 0] = i # row
                coordinates[i*num_patch+j, 1] = j # col
        self.coordinates = coordinates.unsqueeze(0)

        # initialize weight
        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches ** .5),
                                            cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0
        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def generate_raster_permutation(self, N, L):
        """
        Generate raster permutation
        small to large
       """

        width = int(L ** 0.5)
        permutation = torch.zeros((N, width, width))

        init_value = 0
        odd_row = torch.tensor([13 - i for i in range(width)])
        even_row = torch.tensor([i for i in range(width)])
        for i in range(width):
            if i % 2 == 0:
                permutation[:, i, :] = even_row + init_value
            else:
                permutation[:, i, :] = odd_row + init_value

            init_value += width

        # print(permutation)
        permutation = permutation.reshape(N, L)

        return permutation
    
    def generate_center_permutation(self, N, L, center_first=True):
        """
        Generate center-out permutation
        small to large
       """

        width = int(L ** 0.5)
        half_width = width // 2
        permutation = torch.rand((N, width, width))

        if center_first:
            # center 6-7: (-3, -2)
            permutation[:, half_width-1:half_width+1, half_width-1:half_width+1] -= 1
            # surrounding 4-9 (-2 -1)
            permutation[:, half_width-3:half_width+3, half_width-3:half_width+3] -= 1
            # surrounding 2-11 (-1 -0)
            permutation[:, half_width-5:half_width+5, half_width-5:half_width+5] -= 1
            # surrounding 0-13 (0 1)
            # permutation[:, half_width-7:half_width+7, half_width-7:half_width+7] -= 1
        else:
            # center 6-7: (-3, -2)
            permutation[:, half_width-1:half_width+1, half_width-1:half_width+1] += 1
            # surrounding 4-9 (-2 -1)
            permutation[:, half_width-3:half_width+3, half_width-3:half_width+3] += 1
            # surrounding 2-11 (-1 -0)
            permutation[:, half_width-5:half_width+5, half_width-5:half_width+5] += 1
            # surrounding 0-13 (0 1)
            # permutation[:, half_width-7:half_width+7, half_width-7:half_width+7] += 1

        permutation = permutation.reshape(N, L)

        return permutation

    def generate_stochastic_center_permutation(self, N, L):
        """
        Generate stochastic center permutation
        small to large
       """

        width = int(L ** 0.5)
        permutation = torch.rand((N, width, width))

        center_row, center_col = torch.rand((N)) * (width - 1), torch.rand((N)) * (width - 1)

        for i in range(N):
            row_split = self.split_matrix[int(center_row[i]), :, :] # 2x4
            col_split = self.split_matrix[int(center_col[i]), :, :] # 2x4
            for j in range(3):
                permutation[i, int(row_split[0][j]):int(row_split[1][j]), int(col_split[0][j]):int(col_split[1][j])] -= 1

        permutation = permutation.reshape(N, L)
        return permutation
    
    def generate_guided_center_permutation(self, attention_maps):
        """
        Generate attention guided center permutation
        small to large
       """

        N, L = attention_maps.shape
        width = int(L ** 0.5)
        permutation = torch.rand((N, width, width))

        _, max_index = torch.max(attention_maps, dim=-1)
        center_row, center_col = max_index // width, max_index % width
        # attention_maps = attention_maps.reshape(N, width, width)

        for i in range(N):
            row_split = self.split_matrix[center_row[i], :, :] # 2x4
            col_split = self.split_matrix[center_col[i], :, :] # 2x4
            for j in range(3):
                permutation[i, int(row_split[0][j]):int(row_split[1][j]), int(col_split[0][j]):int(col_split[1][j])] -= 1

        permutation = permutation.reshape(N, L)
        return permutation

    def generate_attention_distance_center_permutation(self, attention_maps):
        """
        Generate attention guided gaussian center permutation
        small to large
       """

        N, L = attention_maps.shape
        width = int(L ** 0.5)

        _, max_index = torch.max(attention_maps, dim=-1)
        center_row, center_col = max_index // width, max_index % width

        # smaller distance to center, autoregression first
        self.coordinates = self.coordinates.cuda()
        permutation = (self.coordinates[:, :, 0] - center_row.unsqueeze(1)) ** 2 + (self.coordinates[:, :, 1] - center_col.unsqueeze(1)) ** 2 # N L
        permutation = permutation ** 0.5

        # add randomness for patches with the same distance
        permutation += torch.rand(N, L).cuda() * 1e-3

        return permutation

    def generate_attention_mask(self, x, attention_maps=None):
        """
        Generate permutation mask(content mask and query mask)
       """
        N, L, D = x.shape  # batch, length, dim

        # generate permutation
        if self.permutation_type == 'zigzag':
            permutation = [i for i in range(L)]
            permutation = torch.tensor(permutation).repeat(N, 1).cuda()
        elif self.permutation_type == 'raster':
            permutation = self.generate_raster_permutation(N, L).cuda()
        elif self.permutation_type == 'stochastic':
            permutation = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        elif self.permutation_type == 'stochastic_center':
            permutation = self.generate_stochastic_center_permutation(N, L).cuda()
        elif self.permutation_type == 'center2out':
            permutation = self.generate_center_permutation(N, L).cuda()
        elif self.permutation_type == 'attention':
            assert attention_maps != None
            assert attention_maps.shape[1] == L
            permutation = 1 - attention_maps

        elif self.permutation_type == 'attention_guided':
            assert attention_maps != None
            assert attention_maps.shape[1] == L
            permutation = self.generate_guided_center_permutation(attention_maps).cuda()

        elif self.permutation_type == 'attention_center':
            assert attention_maps != None
            assert attention_maps.shape[1] == L
            permutation = self.generate_attention_distance_center_permutation(attention_maps)
        else:
            print("Not supported permutation type!")

        # content mask
        full_mask = torch.full((N, L, L), -math.inf, device=x.device)
        no_mask = torch.zeros((N, L, L), device=x.device)
        mask_h = torch.where(permutation.unsqueeze(-1) < permutation.unsqueeze(1), full_mask, no_mask)  # broadcast-->N*L*L

        # query mask
        mask_g = torch.where(permutation.unsqueeze(-1) <= permutation.unsqueeze(1), full_mask, no_mask)

        # consider cls_token
        top_padding = torch.full((N, 1, L), -math.inf, device=x.device)  # cls token can't see other tokens
        left_padding = torch.zeros((N, L + 1, 1), device=x.device)  # other tokens can see cls token
        mask_h = torch.cat((top_padding, mask_h), dim=1)
        mask_h = torch.cat((left_padding, mask_h), dim=2)
        mask_g = torch.cat((top_padding, mask_g), dim=1)
        mask_g = torch.cat((left_padding, mask_g), dim=2)
        return mask_h.unsqueeze(1), mask_g.unsqueeze(1), permutation

    def forward_aim(self, x, attention_maps=None):

        # embed patches
        x = self.patch_embed(x)

        mask_h, mask_g, permutation = self.generate_attention_mask(x, attention_maps)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]
        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        # permutation mask
        h = x
        g = self.pos_embed.expand(x.shape[0], -1, -1)  # use fixed pos-embedding, not learnable tensor
        for i in range(self.depth):
            h = self.blocks[i](h, mask=mask_h)
            if (i + 1) % self.step == 0:
                g = self.query_blocks[i // self.step](g, h, mask=mask_g)
        g = self.norm(g)
        g = self.prediction_head(g)

        return g, permutation

    def forward_aim_no_mask(self, x, attention_maps=None):

        # embed patches
        x = self.patch_embed(x)

        mask_h, mask_g, permutation = self.generate_attention_mask(x, attention_maps)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]
        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        # permutation mask
        h = x
        g = self.pos_embed.expand(x.shape[0], -1, -1)  # use fixed pos-embedding, not learnable tensor
        for i in range(self.depth):
            h = self.blocks[i](h)
            if (i + 1) % self.step == 0:
                g = self.query_blocks[i // self.step](g, h)
        g = self.norm(g)
        g = self.prediction_head(g)

        return g, permutation


    ############################ for generate
    def generate_raster_permutation_for_generate(self, N, L):
        """
        Generate raster permutation
        small to large
       """

        width = int(L ** 0.5)
        permutation = torch.zeros((N, width, width))

        init_value = 0
        odd_row = torch.tensor([13 - i for i in range(width)])
        even_row = torch.tensor([i for i in range(width)])
        for i in range(width):
            if i < width // 2:
                continue
            if i % 2 == 0:
                permutation[:, i, :] = even_row + init_value
            else:
                permutation[:, i, :] = odd_row + init_value

            init_value += width

        # print(permutation)
        permutation = permutation.reshape(N, L)

        return permutation

    def generate_center_permutation_for_generate(self, N, L):
        """
        Generate center-out permutation
        small to large
       """

        width = int(L ** 0.5)
        half_width = width // 2
        permutation = torch.zeros((N, width, width))

        # center 6-7: (-3, -2)
        # permutation[:, half_width-1:half_width+1, half_width-1:half_width+1] -= 1
        # surrounding 4-9 (-2 -1)
        permutation[:, half_width-3:half_width+3, half_width-3:half_width+3] -= 1
        # surrounding 2-11 (-1 -0)
        permutation[:, half_width-5:half_width+5, half_width-5:half_width+5] -= 1
        # surrounding 0-13 (0 1)
        permutation[:, half_width-7:half_width+7, half_width-7:half_width+7] -= 1
        

        permutation = permutation.reshape(N, L)

        return permutation


    def generate_attention_mask_for_generate(self, x):
        """
        Generate permutation mask(content mask and query mask)
       """
        N, L, D = x.shape  # batch, length, dim

        # generate permutation
        if self.permutation_type == 'raster':
            permutation = self.generate_raster_permutation_for_generate(N, L).cuda()
        elif self.permutation_type == 'center2out':
            permutation = self.generate_center_permutation_for_generate(N, L).cuda()
        else:
            print("Not supported permutation type!")

        # content mask
        full_mask = torch.full((N, L, L), -math.inf, device=x.device)
        no_mask = torch.zeros((N, L, L), device=x.device)

        # query mask
        mask_g = torch.where(permutation.unsqueeze(-1) < permutation.unsqueeze(1), full_mask, no_mask)

        # consider cls_token
        top_padding = torch.zeros((N, 1, L), device=x.device)  # cls token can't see other tokens
        left_padding = torch.zeros((N, L + 1, 1), device=x.device)  # other tokens can see cls token
        mask_g = torch.cat((top_padding, mask_g), dim=1)
        mask_g = torch.cat((left_padding, mask_g), dim=2)
        return mask_g.unsqueeze(1), permutation

    def forward_aim_for_generate(self, x):

        # embed patches
        x = self.patch_embed(x)

        mask_g, permutation = self.generate_attention_mask_for_generate(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # take up half
        # x = x[:, :98, :]

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # encoder
        h = x
        g = self.pos_embed.expand(x.shape[0], -1, -1)
        for i in range(self.depth):
            h = self.blocks[i](h)
            if (i + 1) % self.step == 0:
                g = self.query_blocks[i // self.step](g, h, mask=mask_g)
        g = self.norm(g)
        g = self.prediction_head(g)

        return g, permutation

    def forward_encoder(self, x):
        # embed patches
        x = self.patch_embed(x)
        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]
        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        for i in range(len(self.blocks)-1):
            x = self.blocks[i](x)
        
        self_attention = self.blocks[len(self.blocks)-1](x, return_attention=True) # B H N+1 N+1
        self_attention = torch.mean(self_attention, dim=1)[:, 0, 1:] # B N

        x = self.blocks[len(self.blocks)-1](x)
        x = self.norm(x)

        # calculate attention
        if self.attention_type == 'gap':
            feature_attention = self.calculate_attention_gap(x)
        else:
            feature_attention = self.calculate_attention_cls(x)

        return x, feature_attention, self_attention

    def calculate_attention_cls(self, tokens):
        tokens = torch.nn.functional.normalize(tokens, p=2, dim=-1)
        attention = torch.sum(tokens[:, 0, :].unsqueeze(1) * tokens[:, 1:, :], dim=-1)
        
        attention = attention.softmax(dim=1)

        return attention

    def calculate_attention_gap(self, tokens):
        pth_gap = torch.mean(tokens[:, 1:, :], dim=1, keepdim=True)
        pth_gap = torch.nn.functional.normalize(pth_gap, p=2, dim=-1)
        tokens = torch.nn.functional.normalize(tokens, p=2, dim=-1)
        attention = torch.sum(pth_gap * tokens[:, 1:, :], dim=-1)
        
        attention = attention.softmax(dim=1)

        return attention

    def forward_pixel_loss(self, imgs, pred):
        imgs = self.gaussian_blur(imgs)
        target = self.patchify(imgs)
        pred = pred[:, 1:, :]
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** .5
        if self.loss_type == 'L1':
            loss = (pred - target).abs()
        elif self.loss_type == 'L2':
            loss = (pred - target) ** 2
        return loss.mean(), loss.mean(dim=-1)

    def forward_feature_loss(self, feature, pred):
        feature = feature[:, 1:, :]
        pred = pred[:, 1:, :]
        feature = torch.nn.functional.normalize(feature, p=2, dim=-1)
        pred = torch.nn.functional.normalize(pred, p=2, dim=-1)
        loss = ((pred - feature) ** 2).sum(dim=-1)
        return loss.mean(), loss

    def forward(self, imgs, tokens=None, attention_maps=None, forward_encoder=False):
        if forward_encoder:
            enc_tokens, feature_attention, self_attention = self.forward_encoder(imgs)
            return enc_tokens, feature_attention, self_attention

        pred, permutation = self.forward_aim(imgs, attention_maps)

        if self.predict_feature == 'none':
            loss, loss_map = self.forward_pixel_loss(imgs, pred)
        else:
            assert tokens != None
            loss, loss_map = self.forward_feature_loss(tokens, pred)
        return loss, permutation, loss_map

    def forward_for_visilization(self, imgs, attention_maps=None):
        pred, permutation = self.forward_aim(imgs, attention_maps)

        loss, loss_map = self.forward_pixel_loss(imgs, pred)
        imgs_blur = self.gaussian_blur(imgs)
        
        return loss, permutation, pred, imgs_blur


def aim_base(**kwargs):
    return AimViT(patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, **kwargs)

def aim_large(**kwargs):
    return AimViT(patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, **kwargs)

def aim_huge(**kwargs):
    return AimViT(patch_size=16, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, **kwargs)


if __name__ == '__main__':
    torch.manual_seed(2023)
    model = aim_base(img_size=224,  norm_pix_loss=False,
                      permutation_type='attention_center',
                      prediction_head_type='MLP', loss_type='L2',
                      query_depth=12, share_weight=False,
                      gaussian_kernel_size=9, gaussian_sigma=1)
    model.eval()
    x = torch.rand(1, 3, 224, 224)
    print(model(x))
