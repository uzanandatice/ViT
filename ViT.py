from math import sqrt
from einops import rearrange, repeat, pack
import torch
import torch.nn.functional as F
from torch import nn

# classes
class MLP(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )
    def forward(self, x):
        return self.mlp(x)
    

 # Same dimensions for qkv
class MSA(nn.Module):
    def __init__(self, dim, heads = 8, head_dim = 64, dropout = 0, mask_self=False):
        super().__init__()
        total_dim = head_dim *  heads
        self.heads = heads
        self.mask_self = mask_self
        self.scale = sqrt(head_dim)

        self.norm = nn.LayerNorm(dim)
        self.softmax = nn.Softmax(dim = -1)
        self.to_q = nn.Linear(dim, total_dim, bias = False)
        self.to_k = nn.Linear(dim, total_dim, bias = False)
        self.to_v = nn.Linear(dim, total_dim, bias = True)

        self.fc = nn.Sequential(
            nn.Linear(total_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = self.norm(x)
        q, k, v = [self.to_q(x), self.to_k(x), self.to_v(x)]         
        q, k, v = [rearrange(projection, 'b p (h d) -> b h p d', h=self.heads) for projection in [q, k, v]]
        
        R = torch.matmul(q, k.transpose(-1, -2)) / self.scale

        # mask self token relationship
        if self.mask_self is True:
            mask = torch.eye(R.shape[-1], device = R.device, dtype = torch.bool)
            R = R.masked_fill(mask, float('-inf'))

        attn = self.softmax(R)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h p d -> b p (h d)')
        return self.fc(out) 


# Same dimensions for qkv
# For small datasets
class LSA(nn.Module):
    def __init__(self, dim, heads = 8, head_dim = 64, dropout = 0.):
        super().__init__()
        total_dim = head_dim *  heads
        self.heads = heads
        self.temperature = nn.Parameter(torch.tensor(1/sqrt(head_dim)))

        self.norm = nn.LayerNorm(dim)
        self.softmax = nn.Softmax(dim = -1)
        self.to_q = nn.Linear(dim, total_dim, bias = False)
        self.to_k = nn.Linear(dim, total_dim, bias = False)
        self.to_v = nn.Linear(dim, total_dim, bias = True)

        self.fc = nn.Sequential(
            nn.Linear(total_dim, dim),
            nn.Dropout(dropout)
        ) 

    def forward(self, x):
        x = self.norm(x)
        q, k, v = [self.to_q(x), self.to_k(x), self.to_v(x)] 
        q, k, v = [rearrange(projection, 'b p (h d) -> b h p d', h=self.heads) for projection in [q, k, v]]

        R = torch.matmul(q, k.transpose(-1, -2)) * self.temperature

        mask = torch.eye(R.shape[-1], device = R.device, dtype = torch.bool)
        R = R.masked_fill(mask, float('-inf'))

        attn = self.softmax(R)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.fc(out)   


class Permute(nn.Module):
    def forward(self, x):
        return x.permute((0,2,3,1))
    
    
class ToPatchToken(nn.Module):
    def __init__(self, *, dim, patch_size, channels = 3, shift_amount=0):
        super().__init__()
        self.size = patch_size
        self.shift_amount = shift_amount
        self.to_patch_tokens = nn.Sequential(
            nn.Conv2d(in_channels=channels * (shift_amount * 4 + 1), out_channels=dim, kernel_size=patch_size, stride=patch_size),
            Permute(),
            nn.Flatten(start_dim=1, end_dim=2),
            nn.LayerNorm(dim)
        )

    def forward(self, x):
        if self.shift_amount > 0:
            shifts = []
            b = int(self.size // 4)
            base_shifts = ((b, -1 * b, 0, 0), (-1 * b, b, 0, 0), (0, 0, b, -1 * b), (0, 0, -1 * b, b))

            for i in range(1, self.shift_amount + 1):  
                scaled = tuple(tuple(loc * i for loc in tup) for tup in base_shifts)
                shifts.extend(scaled)  

            shifted_x = [F.pad(x, shift) for shift in shifts]
            x, _ = pack([x, *shifted_x], 'b * w h')
       
        return self.to_patch_tokens(x)


class ResidualConnect(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        
    def forward(self, x):
        res = x
        x = self.fn(x)
        x += res
        return x  


class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels = 3, head_dim = 64, dropout = 0., mask_self=False):
        super().__init__()
        image_height, image_width = image_size
        num_patches = (image_height // patch_size) * (image_width // patch_size)

        self.to_patch_embedding = ToPatchToken(dim=dim, patch_size=patch_size, channels=channels)

        # Gaussian weight init
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(dropout)

        encoderLayers = []
        for _ in range(depth):
            encoderLayers.extend([
                ResidualConnect(MSA(dim, heads = heads, head_dim = head_dim, dropout = dropout, mask_self=mask_self)),
                ResidualConnect(MLP(dim, hidden_dim=mlp_dim, dropout=dropout))
                ])

        self.transformer = nn.Sequential(*encoderLayers)

        self.classifier_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes),
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x, _ = pack([cls_tokens, x], 'b * d')
        x += self.pos_embedding
        x = self.dropout(x)

        x = self.transformer(x)

        return self.classifier_head(x[:, 0])
        

class ViT_small(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels = 3, head_dim = 64, dropout = 0., shift_amount=1):
        super().__init__()
        image_height, image_width = image_size
        num_patches = (image_height // patch_size) * (image_width // patch_size)
        self.to_patch_embedding = ToPatchToken(dim = dim, patch_size = patch_size, channels = channels, shift_amount=shift_amount)

        # Gaussian weight init
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(dropout)

        encoderLayers = []
        for _ in range(depth):
            encoderLayers.extend([
                ResidualConnect(LSA(dim, heads = heads, head_dim = head_dim, dropout = dropout)),
                ResidualConnect(MLP(dim, hidden_dim=mlp_dim, dropout=dropout))
                ])

        self.transformer = nn.Sequential(*encoderLayers)

        self.classifier_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes),
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, _, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x, _ = pack([cls_tokens, x], 'b * d')
        x += self.pos_embedding
        x = self.dropout(x)

        x = self.transformer(x)

        return self.classifier_head(x[:, 0])
    

