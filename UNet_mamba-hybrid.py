import os
import glob
import random
import math
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, roc_curve, auc
from einops import rearrange
from scipy.ndimage import distance_transform_edt

# ========== IMPROVED CONFIGURATION ==========
train_img_dir = "/lfs/jsuri.isu/Sanchit_Mamba/Data/ISIC-2017_Training_Data_Augmented"
train_mask_dir = "/lfs/jsuri.isu/Sanchit_Mamba/Data/ISIC-2017_Training_Part1_GroundTruth_Augmented"
val_img_dir = "/lfs/jsuri.isu/Sanchit_Mamba/Data/ISIC-2017_Validation_Data"
val_mask_dir = "/lfs/jsuri.isu/Sanchit_Mamba/Data/ISIC-2017_Validation_Part1_GroundTruth"

IMG_SIZE = 224
BATCH_SIZE = 64
NUM_WORKERS = 8
EPOCHS = 200
WARMUP_EPOCHS = 5
BASE_LR = 1e-4  # IMPROVED: Increased from 5e-5
MIN_LR = 5e-7   # IMPROVED: Decreased for better fine-tuning
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
BASE_RESULTS_DIR = "/lfs/jsuri.isu/Sanchit_Mamba/results"
USE_AMP = True
USE_MULTI_GPU = True

# ========== IMPROVED OPTIMIZATION SETTINGS ==========
SPATIAL_REDUCTION = 1  # IMPROVED: No reduction for better accuracy
STATE_DIM = 64  # IMPROVED: Increased capacity
USE_PARALLEL_SCAN = True
GRADIENT_ACCUMULATION_STEPS = 2  # IMPROVED: For stability
USE_EMA = True  # IMPROVED: Exponential Moving Average
EMA_DECAY = 0.999

# Reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if DEVICE.startswith("cuda"):
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ========== LOGGING UTILITIES ==========

class MetricsLogger:
    """Formatted metrics logging"""
    def __init__(self, log_file=None):
        self.log_file = log_file
        self.header_printed = False
        
    def print_header(self):
        """Print table header"""
        header = "=" * 100
        columns = f"{'EPOCH':<7}{'PHASE':<7}{'LOSS':<11}{'DICE':<11}{'IOU':<11}{'PREC':<11}{'REC':<11}{'ROC_AUC':<11}{'LR':<11}"
        
        print(header)
        print(columns)
        print(header)
        
        if self.log_file:
            with open(self.log_file, 'a') as f:
                f.write(header + '\n')
                f.write(columns + '\n')
                f.write(header + '\n')
        
        self.header_printed = True
    
    def log_metrics(self, epoch, phase, loss, dice, iou, prec, rec, roc_auc, lr=None):
        """Log metrics in formatted table"""
        if not self.header_printed:
            self.print_header()
        
        lr_str = f"{lr:.2e}" if lr is not None else "N/A"
        
        line = f"{epoch:<7}{phase:<7}{loss:<11.4f}{dice:<11.4f}{iou:<11.4f}{prec:<11.4f}{rec:<11.4f}{roc_auc:<11.4f}{lr_str:<11}"
        
        print(line)
        
        if self.log_file:
            with open(self.log_file, 'a') as f:
                f.write(line + '\n')
    
    def log_message(self, message):
        """Log a custom message"""
        print(message)
        if self.log_file:
            with open(self.log_file, 'a') as f:
                f.write(message + '\n')
    
    def print_separator(self):
        """Print separator line"""
        sep = "-" * 100
        print(sep)
        if self.log_file:
            with open(self.log_file, 'a') as f:
                f.write(sep + '\n')


# ========== IMPROVED DATASET WITH BETTER AUGMENTATION ==========
class ImprovedSegmentationDataset(Dataset):
    def __init__(self, img_dir, mask_dir, img_size=224, exts=("jpg", "png"), augment=False):
        self.img_paths = []
        for ext in exts:
            self.img_paths.extend(sorted(
                [p for p in glob.glob(os.path.join(img_dir, f"*.{ext}")) if "_superpixels" not in p]
            ))

        self.mask_paths = []
        for ext in exts:
            self.mask_paths.extend(sorted(
                [p for p in glob.glob(os.path.join(mask_dir, f"*.{ext}")) if "_superpixels" not in p]
            ))

        assert len(self.img_paths) == len(self.mask_paths), \
            f"Images ({len(self.img_paths)}) and masks ({len(self.mask_paths)}) count mismatch!"

        self.img_size = img_size
        self.augment = augment

        # Base transforms
        self.transform_img = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.transform_mask = transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ])
        
        # IMPROVED: Better augmentation
        if augment:
            self.aug_geometric = transforms.RandomApply([
                transforms.RandomAffine(
                    degrees=20, 
                    translate=(0.1, 0.1), 
                    scale=(0.9, 1.1),
                    shear=10
                )
            ], p=0.5)
            
            self.aug_color = transforms.ColorJitter(
                brightness=0.2, 
                contrast=0.2, 
                saturation=0.2,
                hue=0.1
            )

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert("RGB")
        mask = Image.open(self.mask_paths[idx]).convert("L")

        if self.augment:
            # Random horizontal flip
            if random.random() > 0.5:
                img = transforms.functional.hflip(img)
                mask = transforms.functional.hflip(mask)
            
            # Random vertical flip
            if random.random() > 0.5:
                img = transforms.functional.vflip(img)
                mask = transforms.functional.vflip(mask)
            
            # Geometric augmentation (same seed for img and mask)
            if random.random() > 0.5:
                seed = np.random.randint(2147483647)
                
                random.seed(seed)
                torch.manual_seed(seed)
                img = self.aug_geometric(img)
                
                random.seed(seed)
                torch.manual_seed(seed)
                mask = self.aug_geometric(mask)
            
            # Color augmentation (only for image)
            if random.random() > 0.5:
                img = self.aug_color(img)

        img = self.transform_img(img)
        mask = self.transform_mask(mask)
        mask = (mask > 0.5).float()

        return img, mask


# ========== IMPROVED HYBRID LOSS FUNCTIONS ==========

class FocalTverskyLoss(nn.Module):
    """Focal Tversky Loss for handling class imbalance and hard examples"""
    def __init__(self, alpha=0.3, beta=0.7, gamma=1.33, smooth=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth
    
    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        
        # Flatten
        inputs_flat = inputs.view(-1)
        targets_flat = targets.view(-1)
        
        # True Positives, False Positives & False Negatives
        TP = (inputs_flat * targets_flat).sum()
        FP = ((1 - targets_flat) * inputs_flat).sum()
        FN = (targets_flat * (1 - inputs_flat)).sum()
        
        Tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        FocalTversky = (1 - Tversky) ** self.gamma
        
        return FocalTversky


class BoundaryLoss(nn.Module):
    """Boundary Loss for precise edge detection"""
    def __init__(self, theta0=3, theta=5):
        super().__init__()
        self.theta0 = theta0
        self.theta = theta
    
    def compute_sdf(self, mask):
        """Compute signed distance function"""
        mask_np = mask.cpu().numpy()
        
        normalized_sdf = np.zeros_like(mask_np)
        
        for b in range(mask_np.shape[0]):
            for c in range(mask_np.shape[1]):
                posmask = mask_np[b, c].astype(bool)
                
                if posmask.any():
                    negmask = ~posmask
                    posdis = distance_transform_edt(posmask)
                    negdis = distance_transform_edt(negmask)
                    boundary = posdis + negdis
                    
                    sdf = negdis - posdis
                    sdf = sdf / (boundary.max() + 1e-8)
                    normalized_sdf[b, c] = sdf
        
        return torch.from_numpy(normalized_sdf).float().to(mask.device)
    
    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        
        # Compute signed distance transform
        with torch.no_grad():
            gt_sdf = self.compute_sdf(targets)
        
        # Boundary loss
        boundary_loss = torch.mean(inputs * gt_sdf)
        
        return boundary_loss


class HybridSegmentationLoss(nn.Module):
    """
    Hybrid Loss combining:
    - Focal Tversky (handles imbalance + hard examples)
    - Boundary Loss (precise edges)
    - Dice Loss (region overlap)
    
    BEST FOR SKIN LESION SEGMENTATION WITH YOUR MAMBA MODEL
    """
    def __init__(self, 
                 focal_tversky_weight=0.5,
                 dice_weight=0.3,
                 boundary_weight=0.2,
                 alpha=0.3,  # Tversky alpha
                 beta=0.7,   # Tversky beta
                 gamma=1.33): # Focal gamma
        super().__init__()
        self.focal_tversky_weight = focal_tversky_weight
        self.dice_weight = dice_weight
        self.boundary_weight = boundary_weight
        
        self.focal_tversky = FocalTverskyLoss(alpha, beta, gamma)
        self.boundary = BoundaryLoss()
    
    def dice_loss(self, inputs, targets, smooth=1.0):
        inputs = torch.sigmoid(inputs)
        inputs_flat = inputs.view(-1)
        targets_flat = targets.view(-1)
        
        intersection = (inputs_flat * targets_flat).sum()
        dice = (2. * intersection + smooth) / (inputs_flat.sum() + targets_flat.sum() + smooth)
        
        return 1 - dice
    
    def forward(self, inputs, targets):
        focal_tversky_loss = self.focal_tversky(inputs, targets)
        dice_loss = self.dice_loss(inputs, targets)
        boundary_loss = self.boundary(inputs, targets)
        
        # Check for NaN
        if torch.isnan(focal_tversky_loss):
            focal_tversky_loss = torch.tensor(0.0, device=inputs.device)
        if torch.isnan(dice_loss):
            dice_loss = torch.tensor(0.0, device=inputs.device)
        if torch.isnan(boundary_loss):
            boundary_loss = torch.tensor(0.0, device=inputs.device)
        
        total_loss = (self.focal_tversky_weight * focal_tversky_loss +
                     self.dice_weight * dice_loss +
                     self.boundary_weight * boundary_loss)
        
        return total_loss


# ========== PARALLEL SCAN IMPLEMENTATION ==========

def efficient_parallel_scan(A, B, x):
    """IMPROVED parallel scan with better numerical stability"""
    B_batch, L, D = A.shape
    device = A.device
    dtype = A.dtype
    
    # Clamp A to safe range
    A_safe = torch.clamp(A, min=0.001, max=0.999)
    
    # Log-sum-exp trick for stability
    log_A = torch.log(A_safe + 1e-10)
    log_cum_A = torch.cumsum(log_A, dim=1)
    
    # Avoid overflow
    log_cum_A_max = log_cum_A.max(dim=1, keepdim=True)[0]
    cum_A = torch.exp(log_cum_A - log_cum_A_max)
    
    # Weighted inputs
    BX = B * x
    
    # Prepare padded cumulative A
    cum_A_padded = torch.cat([
        torch.ones(B_batch, 1, D, device=device, dtype=dtype),
        cum_A
    ], dim=1)
    
    log_cum_A_padded = torch.cat([
        torch.zeros(B_batch, 1, D, device=device, dtype=dtype),
        log_cum_A - log_cum_A_max
    ], dim=1)
    
    # Normalize and accumulate
    BX_normalized = BX * torch.exp(-log_cum_A_padded[:, :-1])
    cum_BX_norm = torch.cumsum(BX_normalized, dim=1)
    h = cum_BX_norm * cum_A
    
    return h


# ========== IMPROVED SELECTIVE SSM ==========

class ImprovedSelectiveSSM(nn.Module):
    """IMPROVED: Better initialization and wider ranges"""
    def __init__(self, dim, state_dim=64, use_parallel=True):
        super().__init__()
        self.dim = dim
        self.state_dim = state_dim
        self.use_parallel = use_parallel
        
        # Input projections
        self.x_proj = nn.Linear(dim, state_dim)
        
        # IMPROVED: Softplus for better range
        self.dt_proj = nn.Sequential(
            nn.Linear(dim, state_dim),
            nn.Softplus()
        )
        
        # IMPROVED: Wider A initialization
        A_init = -torch.rand(state_dim) * 0.8 - 0.2  # Range: [-1.0, -0.2]
        self.A_log = nn.Parameter(torch.log(-A_init))
        
        self.B = nn.Linear(dim, state_dim, bias=False)
        self.C = nn.Linear(dim, state_dim, bias=False)
        self.D = nn.Parameter(torch.zeros(dim))
        
        self.out_proj = nn.Linear(state_dim, dim)
        self.norm = nn.LayerNorm(dim)
        
        self._init_weights()
    
    def _init_weights(self):
        # IMPROVED: Larger initialization
        nn.init.xavier_uniform_(self.x_proj.weight, gain=0.5)
        nn.init.xavier_uniform_(self.B.weight, gain=0.4)
        nn.init.xavier_uniform_(self.C.weight, gain=0.4)
        nn.init.xavier_uniform_(self.out_proj.weight, gain=0.5)
        
        nn.init.zeros_(self.x_proj.bias)
        nn.init.zeros_(self.out_proj.bias)
        nn.init.zeros_(self.dt_proj[0].bias)
    
    def forward(self, x):
        B, L, D = x.shape
        
        x_norm = self.norm(x)
        
        # IMPROVED: Better delta scaling
        delta_raw = self.dt_proj(x_norm)
        delta = delta_raw * 0.01 + 1e-4
        
        B_t = self.B(x_norm)
        C_t = self.C(x_norm)
        
        A = -torch.exp(self.A_log)
        
        x_proj = self.x_proj(x_norm)
        
        # IMPROVED: Less aggressive clamping
        deltaA = delta * A.unsqueeze(0).unsqueeze(0)
        deltaA_clamped = torch.clamp(deltaA, min=-10.0, max=-0.0001)
        
        A_discrete = torch.exp(deltaA_clamped)
        B_discrete = delta * B_t
        
        if self.use_parallel:
            h = efficient_parallel_scan(A_discrete, B_discrete, x_proj)
            y = C_t * h
        else:
            h = torch.zeros(B, self.state_dim, device=x.device, dtype=x.dtype)
            outputs = []
            for t in range(L):
                h = A_discrete[:, t] * h + B_discrete[:, t] * x_proj[:, t]
                y_t = C_t[:, t] * h
                outputs.append(y_t)
            y = torch.stack(outputs, dim=1)
        
        if torch.isnan(y).any() or torch.isinf(y).any():
            y = torch.zeros_like(y)
        
        y = self.out_proj(y)
        
        # IMPROVED: Wider D range
        D_safe = torch.clamp(self.D, min=-2.0, max=2.0)
        y = y + D_safe * x
        
        return y


# ========== MAMBA BLOCK ==========

class MambaBlock(nn.Module):
    """Mamba block with improved SSM"""
    def __init__(self, dim, state_dim=64, dropout_p=0.1, spatial_reduction=1):
        super().__init__()
        self.dim = dim
        self.reduction = spatial_reduction
        
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        # Local feature extraction
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False)
        self.bn = nn.BatchNorm2d(dim)
        
        # IMPROVED: Use ImprovedSelectiveSSM
        self.ssm = ImprovedSelectiveSSM(dim, state_dim, use_parallel=USE_PARALLEL_SCAN)
        
        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout_p)
        )
        
        # IMPROVED: Better initialization for learnable scaling
        self.alpha_local = nn.Parameter(torch.ones(1) * 0.5)
        self.alpha_ssm = nn.Parameter(torch.ones(1) * 0.5)
        self.alpha_mlp = nn.Parameter(torch.ones(1) * 0.5)
        
        # Spatial reduction (now 1x by default)
        if spatial_reduction > 1:
            self.downsample = nn.AvgPool2d(spatial_reduction)
            self.upsample = nn.Upsample(scale_factor=spatial_reduction, mode='bilinear', align_corners=False)
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.dwconv.weight, gain=0.5)
        
        for module in self.mlp.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x):
        B, C, H, W = x.shape
        identity = x
        
        # Local features
        x_local = self.bn(self.dwconv(x))
        
        # Spatial reduction (if needed)
        if self.reduction > 1:
            x_reduced = self.downsample(x)
            H_r, W_r = x_reduced.shape[2:]
        else:
            x_reduced = x
            H_r, W_r = H, W
        
        # SSM processing
        x_seq = rearrange(x_reduced, 'b c h w -> b (h w) c')
        x_seq = self.norm1(x_seq)
        x_seq = self.ssm(x_seq)
        x_ssm = rearrange(x_seq, 'b (h w) c -> b c h w', h=H_r, w=W_r)
        
        # Upsample back (if needed)
        if self.reduction > 1:
            x_ssm = self.upsample(x_ssm)
        
        # First residual
        x = identity + self.alpha_local * x_local + self.alpha_ssm * x_ssm
        
        # MLP branch
        x_mlp = rearrange(x, 'b c h w -> b (h w) c')
        x_mlp = self.norm2(x_mlp)
        x_mlp = self.mlp(x_mlp)
        x = x + self.alpha_mlp * rearrange(x_mlp, 'b (h w) c -> b c h w', h=H, w=W)
        
        return x


# ========== SKIP CONNECTION ==========

class SkipConnectionFusion(nn.Module):
    """Skip connection fusion with Mamba processing"""
    def __init__(self, channels, use_mamba=True):
        super().__init__()
        self.use_mamba = use_mamba
        
        if use_mamba:
            self.mamba = MambaBlock(channels, state_dim=STATE_DIM, dropout_p=0.0, spatial_reduction=SPATIAL_REDUCTION)
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid()
        )
        
        # Fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, decoder_feat, encoder_feat):
        if self.use_mamba:
            encoder_feat = self.mamba(encoder_feat)
        
        # Match dimensions
        if decoder_feat.shape[2:] != encoder_feat.shape[2:]:
            decoder_feat = F.interpolate(
                decoder_feat, 
                size=encoder_feat.shape[2:], 
                mode='bilinear', 
                align_corners=False
            )
        
        # Concatenate and fuse
        cat_feat = torch.cat([decoder_feat, encoder_feat], dim=1)
        att = self.attention(cat_feat)
        fused = self.fusion(cat_feat) * att + encoder_feat * (1 - att)
        
        return fused


# ========== ENCODER & DECODER ==========

class HybridEncoder(nn.Module):
    """Hybrid encoder"""
    def __init__(self, in_ch=64, base_ch=64, depths=[2, 2, 2, 2]):
        super().__init__()
        
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, 7, stride=1, padding=3),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(inplace=True)
        )
        
        self.stages = nn.ModuleList()
        ch = base_ch
        
        for stage_idx, depth in enumerate(depths):
            stage_blocks = nn.ModuleList()
            
            for block_idx in range(depth):
                if stage_idx < 2:
                    # CNN blocks
                    block = nn.Sequential(
                        nn.Conv2d(ch, ch, 3, padding=1),
                        nn.BatchNorm2d(ch),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(ch, ch, 3, padding=1),
                        nn.BatchNorm2d(ch),
                    )
                else:
                    # Mamba blocks
                    block = MambaBlock(ch, state_dim=STATE_DIM, spatial_reduction=SPATIAL_REDUCTION)
                
                stage_blocks.append(block)
            
            # Downsampling
            if stage_idx < len(depths) - 1:
                downsample = nn.Sequential(
                    nn.Conv2d(ch, ch * 2, 3, stride=2, padding=1),
                    nn.BatchNorm2d(ch * 2),
                    nn.ReLU(inplace=True)
                )
                next_ch = ch * 2
            else:
                downsample = nn.Identity()
                next_ch = ch
            
            self.stages.append(nn.ModuleDict({
                'blocks': stage_blocks,
                'downsample': downsample
            }))
            
            ch = next_ch
        
        self.feature_channels = [base_ch * (2**i) for i in range(len(depths))]
    
    def forward(self, x):
        features = []
        
        x = self.stem(x)
        features.append(x)
        
        for stage_idx, stage in enumerate(self.stages[:-1]):
            for block in stage['blocks']:
                if stage_idx < 2:
                    x = F.relu(x + block(x))
                else:
                    x = block(x)
            x = stage['downsample'](x)
            features.append(x)
        
        # Last stage
        for block in self.stages[-1]['blocks']:
            x = block(x)
        
        return features


class ImprovedDecoder(nn.Module):
    """Decoder with skip connections"""
    def __init__(self, feature_channels, use_mamba_skip=True):
        super().__init__()
        self.depth = len(feature_channels)
        
        self.up_layers = nn.ModuleList()
        self.skip_fusions = nn.ModuleList()
        
        for i in range(self.depth - 1, 0, -1):
            in_ch = feature_channels[i]
            out_ch_up = feature_channels[i-1]
            
            self.up_layers.append(nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch_up, 2, stride=2),
                nn.BatchNorm2d(out_ch_up),
                nn.ReLU(inplace=True)
            ))
            
            self.skip_fusions.append(
                SkipConnectionFusion(out_ch_up, use_mamba=use_mamba_skip)
            )
        
        self.final = nn.Conv2d(feature_channels[0], 1, 1)
    
    def forward(self, features):
        x = features[-1]
        
        for i, (up, fusion) in enumerate(zip(self.up_layers, self.skip_fusions)):
            x = up(x)
            skip_idx = -(i + 2)
            x = fusion(x, features[skip_idx])
        
        out = self.final(x)
        return out


class MambaUNet(nn.Module):
    """Complete MambaUNet"""
    def __init__(self, in_channels=3, base_channels=32):
        super().__init__()
        
        self.msi = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        
        self.encoder = HybridEncoder(
            in_ch=base_channels, 
            base_ch=base_channels,
            depths=[2, 2, 2, 2]
        )
        
        self.decoder = ImprovedDecoder(
            feature_channels=self.encoder.feature_channels,
            use_mamba_skip=True
        )
    
    def forward(self, x):
        x = self.msi(x)
        features = self.encoder(x)
        out = self.decoder(features)
        
        if out.shape[2:] != (IMG_SIZE, IMG_SIZE):
            out = F.interpolate(out, size=(IMG_SIZE, IMG_SIZE), mode='bilinear', align_corners=False)
        
        return out


# ========== EXPONENTIAL MOVING AVERAGE ==========

class ModelEMA:
    """Exponential Moving Average for model weights"""
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = self.decay * self.shadow[name] + \
                                   (1 - self.decay) * param.data
    
    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]


# ========== COSINE ANNEALING WITH WARM RESTARTS ==========

class CosineAnnealingWarmRestarts:
    """Cosine annealing with warm restarts"""
    def __init__(self, optimizer, T_0=30, T_mult=2, eta_min=1e-6):
        self.optimizer = optimizer
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.T_cur = 0
        self.T_i = T_0
        self.epoch = 0
    
    def step(self):
        self.epoch += 1
        self.T_cur += 1
        
        if self.T_cur >= self.T_i:
            self.T_cur = 0
            self.T_i *= self.T_mult
        
        for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            param_group['lr'] = self.eta_min + (base_lr - self.eta_min) * \
                (1 + math.cos(math.pi * self.T_cur / self.T_i)) / 2
        
        return self.optimizer.param_groups[0]['lr']


# ========== TRAINING UTILITIES ==========

def dice_score(logits, target, threshold=0.5):
    probs = torch.sigmoid(logits)
    pred_bin = (probs > threshold).float()
    tgt = (target > 0.5).float()
    inter = (pred_bin * tgt).sum(dim=(1,2,3))
    union = pred_bin.sum(dim=(1,2,3)) + tgt.sum(dim=(1,2,3))
    dice = (2 * inter / (union + 1e-8)).mean().item()
    return dice


def iou_score(pred, target, threshold=0.5):
    pred_bin = (pred > threshold).float()
    tgt = (target > 0.5).float()
    inter = (pred_bin * tgt).sum(dim=(1,2,3))
    union = (pred_bin + tgt).clamp(0,1).sum(dim=(1,2,3))
    return (inter / (union + 1e-8)).mean().item()


def precision_score(pred, target, threshold=0.5):
    pred_bin = (pred > threshold).float()
    tgt = (target > 0.5).float()
    tp = (pred_bin * tgt).sum(dim=(1,2,3))
    fp = (pred_bin * (1 - tgt)).sum(dim=(1,2,3))
    return (tp / (tp + fp + 1e-8)).mean().item()


def recall_score(pred, target, threshold=0.5):
    pred_bin = (pred > threshold).float()
    tgt = (target > 0.5).float()
    tp = (pred_bin * tgt).sum(dim=(1,2,3))
    fn = ((1 - pred_bin) * tgt).sum(dim=(1,2,3))
    return (tp / (tp + fn + 1e-8)).mean().item()


def save_side_by_side_images(imgs, masks, preds, save_dir, prefix="epoch", max_samples=8):
    os.makedirs(save_dir, exist_ok=True)
    imgs = imgs.cpu()
    masks = masks.cpu()
    preds = preds.cpu()
    n = min(imgs.shape[0], max_samples)
    
    for i in range(n):
        im = imgs[i].permute(1,2,0).numpy()
        im = (im * np.array([0.229,0.224,0.225]) + np.array([0.485,0.456,0.406]))
        im = (im*255).clip(0,255).astype(np.uint8)
        mask = (masks[i].squeeze(0).numpy()*255).astype(np.uint8)
        pred = (preds[i].squeeze(0).numpy()*255).astype(np.uint8)
        
        im_p = Image.fromarray(im)
        mask_p = Image.fromarray(mask).convert("RGB")
        pred_p = Image.fromarray(pred).convert("RGB")
        combined = Image.new('RGB', (im_p.width*3, im_p.height))
        combined.paste(im_p, (0,0))
        combined.paste(mask_p, (im_p.width,0))
        combined.paste(pred_p, (im_p.width*2,0))
        combined.save(os.path.join(save_dir, f"{prefix}_sample_{i+1}.png"))


# ========== IMPROVED TRAINING FUNCTION ==========

def train_one_epoch_improved(model, loader, optimizer, criterion, device, scaler, epoch, ema=None):
    """IMPROVED: Training with gradient accumulation and EMA"""
    model.train()
    losses = []
    
    optimizer.zero_grad()
    
    for batch_idx, (imgs, masks) in enumerate(loader):
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        
        with torch.amp.autocast('cuda', enabled=USE_AMP):
            preds = model(imgs)
            loss = criterion(preds, masks)
            loss = loss / GRADIENT_ACCUMULATION_STEPS
        
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"[Epoch {epoch}] Warning: NaN/Inf loss in batch {batch_idx}, skipping")
            optimizer.zero_grad()
            continue
        
        scaler.scale(loss).backward()
        
        # Update every N steps
        if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            # Update EMA
            if ema is not None:
                ema.update()
        
        losses.append(loss.item() * GRADIENT_ACCUMULATION_STEPS)
    
    # Handle remaining batches
    if len(loader) % GRADIENT_ACCUMULATION_STEPS != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        
        if ema is not None:
            ema.update()
    
    if len(losses) == 0:
        return float('nan')
    
    return float(np.mean(losses))


def evaluate_model(model, loader, criterion, device):
    """Evaluation loop"""
    model.eval()
    losses = []
    dices, ious, precs, recs = [], [], [], []
    all_probs, all_labels = [], []
    
    with torch.no_grad():
        for imgs, masks in loader:
            imgs = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            
            with torch.amp.autocast('cuda', enabled=USE_AMP):
                preds = model(imgs)
                loss = criterion(preds, masks)
            
            if not torch.isnan(loss):
                losses.append(loss.item())
            
            probs = torch.sigmoid(preds)
            dices.append(dice_score(preds, masks))
            ious.append(iou_score(probs, masks))
            precs.append(precision_score(probs, masks))
            recs.append(recall_score(probs, masks))
            
            all_probs.append(probs.cpu().numpy().ravel())
            all_labels.append(masks.cpu().numpy().ravel())
    
    try:
        all_probs_np = np.concatenate(all_probs)
        all_labels_np = np.concatenate(all_labels)
        roc_auc = float(roc_auc_score((all_labels_np > 0.5).astype(int), all_probs_np))
    except:
        roc_auc = float("nan")
    
    metrics = {
        "loss": float(np.mean(losses)) if losses else float('nan'),
        "dice": float(np.mean(dices)),
        "iou": float(np.mean(ious)),
        "precision": float(np.mean(precs)),
        "recall": float(np.mean(recs)),
        "roc_auc": roc_auc
    }
    return metrics, all_probs_np, all_labels_np


def save_roc_curve(y_true, y_probs, save_path):
    """Save ROC curve plot"""
    y_true_bin = (y_true > 0.5).astype(np.uint8)
    fpr, tpr, _ = roc_curve(y_true_bin, y_probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8,8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_training_curves(history, save_dir):
    """Plot training curves"""
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    axes[0,0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0,0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0,0].set_title('Loss', fontsize=14, fontweight='bold')
    axes[0,0].set_xlabel('Epochs')
    axes[0,0].set_ylabel('Loss')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    axes[0,1].plot(epochs, history['train_dice'], 'b-', label='Train Dice', linewidth=2)
    axes[0,1].plot(epochs, history['val_dice'], 'r-', label='Val Dice', linewidth=2)
    axes[0,1].set_title('Dice Score', fontsize=14, fontweight='bold')
    axes[0,1].set_xlabel('Epochs')
    axes[0,1].set_ylabel('Dice')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    axes[1,0].plot(epochs, history['val_iou'], 'g-', label='Val IoU', linewidth=2)
    axes[1,0].set_title('IoU Score', fontsize=14, fontweight='bold')
    axes[1,0].set_xlabel('Epochs')
    axes[1,0].set_ylabel('IoU')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    axes[1,1].plot(epochs, history['learning_rates'], 'purple', linewidth=2)
    axes[1,1].set_title('Learning Rate', fontsize=14, fontweight='bold')
    axes[1,1].set_xlabel('Epochs')
    axes[1,1].set_ylabel('LR')
    axes[1,1].set_yscale('log')
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()


def load_checkpoint_flexible(model, checkpoint_path, device):
    """FIXED: Load checkpoint handling DataParallel wrapper mismatch"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint['model_state_dict']
    
    is_model_parallel = isinstance(model, nn.DataParallel)
    has_module_prefix = any(k.startswith('module.') for k in state_dict.keys())
    
    if is_model_parallel and not has_module_prefix:
        print("   Adding 'module.' prefix to state_dict keys...")
        state_dict = {f'module.{k}': v for k, v in state_dict.items()}
    elif not is_model_parallel and has_module_prefix:
        print("   Removing 'module.' prefix from state_dict keys...")
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict)
    
    return checkpoint


# ========== LEARNING RATE FINDER ==========

def find_optimal_lr(model, train_loader, criterion, device, start_lr=1e-7, end_lr=1e-2, num_iter=100):
    """Find optimal learning rate"""
    print("\n🔍 Running LR Finder...\n")
    
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=start_lr)
    scaler = torch.amp.GradScaler('cuda', enabled=USE_AMP)
    
    lr_mult = (end_lr / start_lr) ** (1 / num_iter)
    losses = []
    lrs = []
    
    data_iter = iter(train_loader)
    for i in range(num_iter):
        try:
            imgs, masks = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            imgs, masks = next(data_iter)
        
        imgs = imgs.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        
        with torch.amp.autocast('cuda', enabled=USE_AMP):
            preds = model(imgs)
            loss = criterion(preds, masks)
        
        if torch.isnan(loss) or torch.isinf(loss):
            continue
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        lrs.append(optimizer.param_groups[0]['lr'])
        losses.append(loss.item())
        
        for param_group in optimizer.param_groups:
            param_group['lr'] *= lr_mult
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(lrs, losses)
    plt.xscale('log')
    plt.xlabel('Learning Rate')
    plt.ylabel('Loss')
    plt.title('Learning Rate Finder')
    plt.grid(True, alpha=0.3)
    plt.savefig('lr_finder.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Find optimal LR
    gradients = np.gradient(losses)
    optimal_idx = np.argmin(gradients)
    optimal_lr = lrs[optimal_idx]
    
    print(f"📊 Suggested optimal LR: {optimal_lr:.2e}")
    print(f"📊 Suggested range: [{optimal_lr/10:.2e}, {optimal_lr*2:.2e}]")
    print(f"📊 Plot saved to: lr_finder.png\n")
    
    return optimal_lr


# ========== MAIN TRAINING FUNCTION ==========

def run_training():
    device = DEVICE
    
    # GPU info
    num_gpus = torch.cuda.device_count()
    print(f"🚀 Starting IMPROVED training on device: {device}")
    print(f"🎮 Available GPUs: {num_gpus}")
    if num_gpus > 1:
        for i in range(num_gpus):
            print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"   Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")
    
    print(f"\n⚡ IMPROVED OPTIMIZATION SETTINGS:")
    print(f"   Spatial Reduction: {SPATIAL_REDUCTION}x (NO REDUCTION)")
    print(f"   State Dimension: {STATE_DIM}")
    print(f"   Parallel Scan: {'✅ ENABLED' if USE_PARALLEL_SCAN else '❌ DISABLED'}")
    print(f"   Gradient Accumulation: {GRADIENT_ACCUMULATION_STEPS} steps")
    print(f"   EMA: {'✅ ENABLED' if USE_EMA else '❌ DISABLED'}")
    
    print(f"\nMixed Precision: {USE_AMP}")
    print(f"Multi-GPU: {USE_MULTI_GPU and num_gpus > 1}")
    print(f"Base Learning Rate: {BASE_LR}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Warmup Epochs: {WARMUP_EPOCHS}")
    
    # IMPROVED: Better dataset with augmentation
    train_dataset = ImprovedSegmentationDataset(train_img_dir, train_mask_dir, img_size=IMG_SIZE, augment=True)
    val_dataset = ImprovedSegmentationDataset(val_img_dir, val_mask_dir, img_size=IMG_SIZE, augment=False)
    
    print(f"\n📊 Train samples: {len(train_dataset)}")
    print(f"📊 Val samples: {len(val_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                             num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                           num_workers=NUM_WORKERS, pin_memory=True)
    
    # Model
    model = MambaUNet(in_channels=3, base_channels=32).to(device)
    
    # Multi-GPU wrapper
    if USE_MULTI_GPU and num_gpus > 1:
        print(f"\n🔥 Wrapping model with DataParallel for {num_gpus} GPUs")
        model = nn.DataParallel(model)
        model_without_parallel = model.module
    else:
        model_without_parallel = model
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"🏗️  Total parameters: {total_params:,}")
    print(f"🏗️  Trainable parameters: {trainable_params:,}")
    
    # IMPROVED: Hybrid Segmentation Loss
    criterion = HybridSegmentationLoss(
        focal_tversky_weight=0.5,
        dice_weight=0.3,
        boundary_weight=0.2,
        alpha=0.3,
        beta=0.7,
        gamma=1.33
    )
    print(f"\n🎯 Using HybridSegmentationLoss:")
    print(f"   Focal Tversky: 0.5")
    print(f"   Dice: 0.3")
    print(f"   Boundary: 0.2")
    
    # Optimizer
    if USE_MULTI_GPU and num_gpus > 1:
        encoder_params = list(model.module.encoder.parameters())
        decoder_params = list(model.module.decoder.parameters())
        msi_params = list(model.module.msi.parameters())
    else:
        encoder_params = list(model.encoder.parameters())
        decoder_params = list(model.decoder.parameters())
        msi_params = list(model.msi.parameters())
    
    optimizer = torch.optim.AdamW([
        {'params': msi_params, 'lr': BASE_LR},
        {'params': encoder_params, 'lr': BASE_LR},
        {'params': decoder_params, 'lr': BASE_LR}
    ], weight_decay=1e-4)
    
    # IMPROVED: Cosine Annealing with Warm Restarts
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=30,
        T_mult=2,
        eta_min=MIN_LR
    )
    print(f"\n📅 Using CosineAnnealingWarmRestarts:")
    print(f"   T_0: 30 epochs")
    print(f"   T_mult: 2")
    print(f"   eta_min: {MIN_LR}")
    
    # IMPROVED: EMA
    ema = None
    if USE_EMA:
        ema = ModelEMA(model, decay=EMA_DECAY)
        print(f"\n💫 EMA enabled with decay: {EMA_DECAY}")
    
    scaler = torch.amp.GradScaler('cuda', enabled=USE_AMP)
    
    # Results directories
    results_dir = os.path.join(BASE_RESULTS_DIR, "mamba_unet_improved")
    seg_dir = os.path.join(results_dir, "segmented_images")
    roc_dir = os.path.join(results_dir, "roc")
    model_dir = os.path.join(results_dir, "models")
    
    for p in [seg_dir, roc_dir, model_dir]:
        os.makedirs(p, exist_ok=True)
    
    # Logger
    log_file = os.path.join(results_dir, "training_log.txt")
    logger = MetricsLogger(log_file=log_file)
    
    best_dice = 0.0
    patience = 30
    patience_counter = 0
    
    history = {
        'train_loss': [], 'train_dice': [], 'val_loss': [], 
        'val_dice': [], 'val_iou': [], 'learning_rates': []
    }
    
    print("\n" + "="*100)
    print("🏋️  STARTING IMPROVED TRAINING")
    print("="*100 + "\n")
    
    logger.print_header()
    
    # Training loop
    for epoch in range(1, EPOCHS + 1):
        # Train with EMA
        train_loss = train_one_epoch_improved(
            model, train_loader, optimizer, criterion, device, scaler, epoch, ema
        )
        
        if np.isnan(train_loss):
            logger.log_message(f"\n❌ Training failed at epoch {epoch}")
            break
        
        # Evaluate on train set
        train_metrics, _, _ = evaluate_model(model, train_loader, criterion, device)
        
        # Evaluate on val set with EMA weights
        if ema is not None:
            ema.apply_shadow()
        
        val_metrics, all_probs, all_labels = evaluate_model(model, val_loader, criterion, device)
        
        if ema is not None:
            ema.restore()
        
        # Update LR
        lr_now = scheduler.step()
        
        # Log metrics
        logger.log_metrics(
            epoch=epoch,
            phase="TRAIN",
            loss=train_metrics['loss'],
            dice=train_metrics['dice'],
            iou=train_metrics['iou'],
            prec=train_metrics['precision'],
            rec=train_metrics['recall'],
            roc_auc=train_metrics['roc_auc'],
            lr=lr_now
        )
        
        logger.log_metrics(
            epoch=epoch,
            phase="VAL",
            loss=val_metrics['loss'],
            dice=val_metrics['dice'],
            iou=val_metrics['iou'],
            prec=val_metrics['precision'],
            rec=val_metrics['recall'],
            roc_auc=val_metrics['roc_auc'],
            lr=None
        )
        
        # Store history
        history['train_loss'].append(train_metrics['loss'])
        history['train_dice'].append(train_metrics['dice'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_dice'].append(val_metrics['dice'])
        history['val_iou'].append(val_metrics['iou'])
        history['learning_rates'].append(lr_now)
        
        # Save best model
        if val_metrics['dice'] > best_dice:
            best_dice = val_metrics['dice']
            patience_counter = 0
            
            best_path = os.path.join(model_dir, "best_model.pth")
            
            # Save with EMA weights
            if ema is not None:
                ema.apply_shadow()
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_without_parallel.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'dice': best_dice,
                'val_metrics': val_metrics,
                'history': history,
                'ema_state_dict': ema.shadow if ema else None
            }, best_path)
            
            if ema is not None:
                ema.restore()
            
            logger.log_message(f"💾 Saved best model (Dice: {best_dice:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.log_message(f"\n⏹️  Early stopping triggered (no improvement for {patience} epochs)")
                break
        
        # Save checkpoint every 20 epochs
        if epoch % 20 == 0:
            checkpoint_path = os.path.join(model_dir, f"checkpoint_epoch_{epoch}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_without_parallel.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'dice': val_metrics['dice'],
                'history': history,
                'ema_state_dict': ema.shadow if ema else None
            }, checkpoint_path)
        
        # Save visualizations every 10 epochs
        if epoch % 10 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                for imgs, masks in val_loader:
                    imgs = imgs.to(device)
                    masks = masks.to(device)
                    preds = model(imgs)
                    save_side_by_side_images(
                        imgs, masks, torch.sigmoid(preds), 
                        seg_dir, prefix=f"epoch_{epoch:03d}"
                    )
                    break
        
        # Save intermediate results
        with open(os.path.join(results_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=2)
        
        plot_training_curves(history, results_dir)
    
    logger.print_separator()
    logger.log_message("🎉 Training completed!")
    logger.print_separator()
    
    # Load best model for final evaluation
    if os.path.exists(os.path.join(model_dir, "best_model.pth")):
        logger.log_message("\n📈 Loading best model for final evaluation...")
        
        checkpoint = load_checkpoint_flexible(
            model, 
            os.path.join(model_dir, "best_model.pth"),
            device
        )
        
        final_metrics, final_probs, final_labels = evaluate_model(
            model, val_loader, criterion, device
        )
        
        logger.log_message(f"\n🏆 FINAL RESULTS (Best Model):")
        logger.log_message(f"   Dice Score:  {final_metrics['dice']:.4f}")
        logger.log_message(f"   IoU Score:   {final_metrics['iou']:.4f}")
        logger.log_message(f"   Precision:   {final_metrics['precision']:.4f}")
        logger.log_message(f"   Recall:      {final_metrics['recall']:.4f}")
        logger.log_message(f"   ROC AUC:     {final_metrics['roc_auc']:.4f}")
        logger.log_message(f"   Best Epoch:  {checkpoint['epoch']}")
        
        # Save final ROC curve
        save_roc_curve(final_labels, final_probs, os.path.join(roc_dir, "final_roc.png"))
        
        # Save final metrics
        final_results = {
            'best_epoch': checkpoint['epoch'],
            'best_dice': best_dice,
            'final_metrics': final_metrics,
            'model_params': total_params,
            'trainable_params': trainable_params,
            'optimization_settings': {
                'spatial_reduction': SPATIAL_REDUCTION,
                'state_dim': STATE_DIM,
                'parallel_scan': USE_PARALLEL_SCAN,
                'gradient_accumulation': GRADIENT_ACCUMULATION_STEPS,
                'ema': USE_EMA,
                'base_lr': BASE_LR,
                'min_lr': MIN_LR
            }
        }
        
        with open(os.path.join(results_dir, "final_results.json"), "w") as f:
            json.dump(final_results, f, indent=2)
        
        logger.log_message(f"\n📁 All results saved to: {results_dir}")
        logger.print_separator()
        
        return model, history, final_metrics
    else:
        logger.log_message("\n⚠️  No best model found")
        return model, history, {}


# ========== RESUME TRAINING FUNCTION ==========

def resume_training_from_checkpoint(checkpoint_path, new_epochs=None):
    """Resume training from a saved checkpoint"""
    device = DEVICE
    
    print(f"📂 Loading checkpoint from: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    start_epoch = checkpoint['epoch']
    
    print(f"✅ Loaded checkpoint from epoch {start_epoch}")
    
    # Create model
    model = MambaUNet(in_channels=3, base_channels=32).to(device)
    
    # Handle multi-GPU
    num_gpus = torch.cuda.device_count()
    if USE_MULTI_GPU and num_gpus > 1:
        print(f"🔥 Using {num_gpus} GPUs with DataParallel")
        model = nn.DataParallel(model)
        model_without_parallel = model.module
    else:
        model_without_parallel = model
    
    # Load model state
    state_dict = checkpoint['model_state_dict']
    is_model_parallel = isinstance(model, nn.DataParallel)
    has_module_prefix = any(k.startswith('module.') for k in state_dict.keys())
    
    if is_model_parallel and not has_module_prefix:
        state_dict = {f'module.{k}': v for k, v in state_dict.items()}
    elif not is_model_parallel and has_module_prefix:
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict)
    
    # Load datasets
    train_dataset = ImprovedSegmentationDataset(train_img_dir, train_mask_dir, img_size=IMG_SIZE, augment=True)
    val_dataset = ImprovedSegmentationDataset(val_img_dir, val_mask_dir, img_size=IMG_SIZE, augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                             num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                           num_workers=NUM_WORKERS, pin_memory=True)
    
    # Optimizer
    if USE_MULTI_GPU and num_gpus > 1:
        encoder_params = list(model.module.encoder.parameters())
        decoder_params = list(model.module.decoder.parameters())
        msi_params = list(model.module.msi.parameters())
    else:
        encoder_params = list(model.encoder.parameters())
        decoder_params = list(model.decoder.parameters())
        msi_params = list(model.msi.parameters())
    
    optimizer = torch.optim.AdamW([
        {'params': msi_params, 'lr': BASE_LR},
        {'params': encoder_params, 'lr': BASE_LR},
        {'params': decoder_params, 'lr': BASE_LR}
    ], weight_decay=1e-4)
    
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Scheduler
    total_epochs = new_epochs if new_epochs else EPOCHS
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=30,
        T_mult=2,
        eta_min=MIN_LR
    )
    scheduler.epoch = start_epoch
    
    # Loss
    criterion = HybridSegmentationLoss(
        focal_tversky_weight=0.5,
        dice_weight=0.3,
        boundary_weight=0.2,
        alpha=0.3,
        beta=0.7,
        gamma=1.33
    )
    
    scaler = torch.amp.GradScaler('cuda', enabled=USE_AMP)
    
    # EMA
    ema = None
    if USE_EMA:
        ema = ModelEMA(model, decay=EMA_DECAY)
        if 'ema_state_dict' in checkpoint and checkpoint['ema_state_dict']:
            ema.shadow = checkpoint['ema_state_dict']
    
    # Results directory
    results_dir = os.path.dirname(os.path.dirname(checkpoint_path))
    seg_dir = os.path.join(results_dir, "segmented_images")
    roc_dir = os.path.join(results_dir, "roc")
    model_dir = os.path.join(results_dir, "models")
    
    log_file = os.path.join(results_dir, "training_log.txt")
    logger = MetricsLogger(log_file=log_file)
    
    history = checkpoint.get('history', {
        'train_loss': [], 'train_dice': [], 'val_loss': [], 
        'val_dice': [], 'val_iou': [], 'learning_rates': []
    })
    
    best_dice = checkpoint.get('dice', 0.0)
    patience_counter = 0
    patience = 30
    
    print(f"\n🔄 Resuming training from epoch {start_epoch + 1} to {total_epochs}")
    print(f"📊 Previous best Dice: {best_dice:.4f}\n")
    
    logger.log_message(f"\n{'='*100}")
    logger.log_message(f"🔄 RESUMED TRAINING FROM EPOCH {start_epoch}")
    logger.log_message(f"{'='*100}\n")
    logger.print_header()
    
    # Continue training loop
    for epoch in range(start_epoch + 1, total_epochs + 1):
        train_loss = train_one_epoch_improved(
            model, train_loader, optimizer, criterion, device, scaler, epoch, ema
        )
        
        if np.isnan(train_loss):
            logger.log_message(f"\n❌ Training failed at epoch {epoch}")
            break
        
        train_metrics, _, _ = evaluate_model(model, train_loader, criterion, device)
        
        if ema is not None:
            ema.apply_shadow()
        
        val_metrics, all_probs, all_labels = evaluate_model(model, val_loader, criterion, device)
        
        if ema is not None:
            ema.restore()
        
        lr_now = scheduler.step()
        
        logger.log_metrics(
            epoch=epoch, phase="TRAIN",
            loss=train_metrics['loss'], dice=train_metrics['dice'], iou=train_metrics['iou'],
            prec=train_metrics['precision'], rec=train_metrics['recall'], 
            roc_auc=train_metrics['roc_auc'], lr=lr_now
        )
        
        logger.log_metrics(
            epoch=epoch, phase="VAL",
            loss=val_metrics['loss'], dice=val_metrics['dice'], iou=val_metrics['iou'],
            prec=val_metrics['precision'], rec=val_metrics['recall'], 
            roc_auc=val_metrics['roc_auc'], lr=None
        )
        
        history['train_loss'].append(train_metrics['loss'])
        history['train_dice'].append(train_metrics['dice'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_dice'].append(val_metrics['dice'])
        history['val_iou'].append(val_metrics['iou'])
        history['learning_rates'].append(lr_now)
        
        if val_metrics['dice'] > best_dice:
            best_dice = val_metrics['dice']
            patience_counter = 0
            
            best_path = os.path.join(model_dir, "best_model.pth")
            
            if ema is not None:
                ema.apply_shadow()
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_without_parallel.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'dice': best_dice,
                'val_metrics': val_metrics,
                'history': history,
                'ema_state_dict': ema.shadow if ema else None
            }, best_path)
            
            if ema is not None:
                ema.restore()
            
            logger.log_message(f"💾 Saved best model (Dice: {best_dice:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.log_message(f"\n⏹️  Early stopping triggered")
                break
        
        if epoch % 20 == 0:
            checkpoint_path_new = os.path.join(model_dir, f"checkpoint_epoch_{epoch}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_without_parallel.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'dice': val_metrics['dice'],
                'history': history,
                'ema_state_dict': ema.shadow if ema else None
            }, checkpoint_path_new)
        
        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                for imgs, masks in val_loader:
                    imgs = imgs.to(device)
                    masks = masks.to(device)
                    preds = model(imgs)
                    save_side_by_side_images(
                        imgs, masks, torch.sigmoid(preds), 
                        seg_dir, prefix=f"epoch_{epoch:03d}"
                    )
                    break
        
        with open(os.path.join(results_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=2)
        
        plot_training_curves(history, results_dir)
    
    logger.print_separator()
    logger.log_message("🎉 Resumed training completed!")
    
    return model, history


# ========== DEBUG AND UTILITY FUNCTIONS ==========

def test_single_batch():
    """Test model on a single batch for debugging"""
    print("🧪 Testing single batch...\n")
    
    device = DEVICE
    
    train_dataset = ImprovedSegmentationDataset(train_img_dir, train_mask_dir, img_size=IMG_SIZE)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=False)
    
    imgs, masks = next(iter(train_loader))
    imgs = imgs.to(device)
    masks = masks.to(device)
    
    print(f"📦 Batch info:")
    print(f"   Images shape: {imgs.shape}")
    print(f"   Masks shape: {masks.shape}")
    print(f"   Images range: [{imgs.min().item():.3f}, {imgs.max().item():.3f}]")
    print(f"   Masks range: [{masks.min().item():.3f}, {masks.max().item():.3f}]")
    
    model = MambaUNet(in_channels=3, base_channels=32).to(device)
    model.eval()
    
    print("\n🔄 Running forward pass...")
    with torch.no_grad():
        preds = model(imgs)
    
    print(f"   Predictions shape: {preds.shape}")
    print(f"   Predictions range: [{preds.min().item():.3f}, {preds.max().item():.3f}]")
    
    if torch.isnan(preds).any():
        print("   ❌ NaN detected in predictions!")
        return False
    elif torch.isinf(preds).any():
        print("   ❌ Inf detected in predictions!")
        return False
    else:
        print("   ✅ Predictions look good!")
    
    print("\n🔄 Computing loss...")
    criterion = HybridSegmentationLoss()
    loss = criterion(preds, masks)
    
    print(f"   Loss: {loss.item():.4f}")
    
    if torch.isnan(loss):
        print("   ❌ Loss is NaN!")
        return False
    else:
        print("   ✅ Loss is valid!")
    
    print("\n🔄 Testing backward pass...")
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    optimizer.zero_grad()
    preds = model(imgs)
    loss = criterion(preds, masks)
    loss.backward()
    
    nan_grads = 0
    total_grads = 0
    for name, param in model.named_parameters():
        if param.grad is not None:
            total_grads += 1
            if torch.isnan(param.grad).any():
                nan_grads += 1
                print(f"   ❌ NaN gradient in: {name}")
    
    if nan_grads > 0:
        print(f"   ❌ Found NaN in {nan_grads}/{total_grads} gradients!")
        return False
    else:
        print(f"   ✅ All {total_grads} gradients are valid!")
    
    optimizer.step()
    
    print("\n✅ Single batch test passed!\n")
    return True


def benchmark_ssm():
    """Compare sequential vs parallel SSM speed"""
    import time
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    B, L, D = 4, 224*224, 64
    state_dim = STATE_DIM
    
    print("\n" + "="*80)
    print("⚡ BENCHMARKING SSM IMPLEMENTATIONS")
    print("="*80)
    print(f"Batch size: {B}")
    print(f"Sequence length: {L:,} (224×224 image)")
    print(f"Feature dim: {D}")
    print(f"State dimension: {state_dim}")
    print()
    
    x = torch.randn(B, L, D).to(device)
    
    # Sequential SSM
    print("Testing SEQUENTIAL SSM...")
    ssm_seq = ImprovedSelectiveSSM(D, state_dim, use_parallel=False).to(device)
    ssm_seq.eval()
    
    with torch.no_grad():
        _ = ssm_seq(x)
        torch.cuda.synchronize()
        start = time.time()
        out_seq = ssm_seq(x)
        torch.cuda.synchronize()
        time_seq = time.time() - start
    
    print(f"   Time: {time_seq:.3f}s\n")
    
    # Parallel SSM
    print("Testing PARALLEL SSM...")
    ssm_par = ImprovedSelectiveSSM(D, state_dim, use_parallel=True).to(device)
    ssm_par.load_state_dict(ssm_seq.state_dict())
    ssm_par.eval()
    
    with torch.no_grad():
        _ = ssm_par(x)
        torch.cuda.synchronize()
        start = time.time()
        out_par = ssm_par(x)
        torch.cuda.synchronize()
        time_par = time.time() - start
    
    print(f"   Time: {time_par:.3f}s\n")
    
    speedup = time_seq / time_par
    print("="*80)
    print(f"🚀 SPEEDUP: {speedup:.1f}x faster!")
    print(f"⏱️  Sequential: {time_seq:.3f}s")
    print(f"⏱️  Parallel:   {time_par:.3f}s")
    print("="*80)
    
    diff = (out_seq - out_par).abs().mean().item()
    print(f"\n📊 Mean absolute difference: {diff:.6f}")
    
    return speedup


def run_debug_mode():
    """Debug mode"""
    print("🔧 DEBUG MODE\n")
    
    device = DEVICE
    
    print("1️⃣ Testing model architecture...")
    model = MambaUNet(in_channels=3, base_channels=32).to(device)
    x = torch.randn(2, 3, IMG_SIZE, IMG_SIZE).to(device)
    
    print(f"   Input shape: {x.shape}")
    
    with torch.no_grad():
        out = model(x)
    
    print(f"   Output shape: {out.shape}")
    print(f"   Output range: [{out.min().item():.3f}, {out.max().item():.3f}]")
    
    if torch.isnan(out).any():
        print("   ❌ NaN detected!")
    else:
        print("   ✅ Model OK\n")
    
    print("2️⃣ Testing data loading...")
    try:
        train_dataset = ImprovedSegmentationDataset(train_img_dir, train_mask_dir, img_size=IMG_SIZE)
        val_dataset = ImprovedSegmentationDataset(val_img_dir, val_mask_dir, img_size=IMG_SIZE)
        
        print(f"   Train samples: {len(train_dataset)}")
        print(f"   Val samples: {len(val_dataset)}")
        
        img, mask = train_dataset[0]
        print(f"   Image shape: {img.shape}, range: [{img.min():.3f}, {img.max():.3f}]")
        print(f"   Mask shape: {mask.shape}, range: [{mask.min():.3f}, {mask.max():.3f}]")
        print("   ✅ Data loading OK\n")
    except Exception as e:
        print(f"   ❌ Data loading failed: {e}\n")
        return
    
    print("3️⃣ Testing loss function...")
    try:
        criterion = HybridSegmentationLoss()
        x = torch.randn(2, 1, 224, 224).to(device)
        y = torch.rand(2, 1, 224, 224).to(device)
        loss = criterion(x, y)
        print(f"   Loss value: {loss.item():.4f}")
        print("   ✅ Loss function OK\n")
    except Exception as e:
        print(f"   ❌ Loss function failed: {e}\n")
        return
    
    print("🎉 All tests passed!\n")


# ========== MAIN ENTRY POINT ==========

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--debug":
            run_debug_mode()
        
        elif sys.argv[1] == "--benchmark":
            benchmark_ssm()
        
        elif sys.argv[1] == "--test-batch":
            test_single_batch()
        
        elif sys.argv[1] == "--lr-finder":
            print("🔍 Running LR Finder...\n")
            device = DEVICE
            
            train_dataset = ImprovedSegmentationDataset(train_img_dir, train_mask_dir, img_size=IMG_SIZE)
            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                                     num_workers=NUM_WORKERS, pin_memory=True)
            
            model = MambaUNet(in_channels=3, base_channels=32).to(device)
            criterion = HybridSegmentationLoss()
            
            optimal_lr = find_optimal_lr(model, train_loader, criterion, device)
            print(f"\n💡 Recommended: Update BASE_LR to {optimal_lr:.2e} in config")
        
        elif sys.argv[1] == "--resume":
            if len(sys.argv) < 3:
                print("Usage: python script.py --resume <checkpoint_path> [new_total_epochs]")
                print("Example: python script.py --resume results/mamba_unet_improved/models/checkpoint_epoch_50.pth 150")
            else:
                checkpoint_path = sys.argv[2]
                new_epochs = int(sys.argv[3]) if len(sys.argv) > 3 else None
                resume_training_from_checkpoint(checkpoint_path, new_epochs)
        
        else:
            print("🚀 IMPROVED MAMBA UNET TRAINING")
            print("\nUsage:")
            print("  python script.py                           # Run full training")
            print("  python script.py --debug                   # Test components")
            print("  python script.py --benchmark               # Benchmark SSM speed")
            print("  python script.py --test-batch              # Test single batch")
            print("  python script.py --lr-finder               # Find optimal learning rate")
            print("  python script.py --resume <ckpt> [epochs]  # Resume training")
    
    else:
        try:
            print("🔍 Verifying data paths...")
            assert os.path.isdir(train_img_dir), f"❌ train_img_dir not found: {train_img_dir}"
            assert os.path.isdir(train_mask_dir), f"❌ train_mask_dir not found: {train_mask_dir}"
            assert os.path.isdir(val_img_dir), f"❌ val_img_dir not found: {val_img_dir}"
            assert os.path.isdir(val_mask_dir), f"❌ val_mask_dir not found: {val_mask_dir}"
            print("✅ All paths verified!\n")
            
            model, history, final_metrics = run_training()
            
        except AssertionError as e:
            print(f"\n❌ ERROR: {e}")
            print("\nPlease check your data paths in the configuration section.")
        except Exception as e:
            print(f"\n❌ UNEXPECTED ERROR: {e}")
            import traceback
            traceback.print_exc()