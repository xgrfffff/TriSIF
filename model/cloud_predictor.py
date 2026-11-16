"""TriSIF stage 1: regional--local fusion for future sky prediction.

Images use [B,T,C,H,W]; solar masks use [B,T,1,H,W] with values in [0,1]. Inputs are considered as normalized and temporally aligned.
CloudPredictor returns (sky, satellite, loss, metrics). 
Train it before passing it to TriSIF.

The spatial encoder/decoder and TAU follow the project's SimVP/OpenSTL implementation. 
The satellite forecast is detached for sky fusion and trained by its own MSE. 
Optional teacher forcing uses a caller-supplied probability; evaluation always uses predictions. 
Example of the original linear schedule:
    progress = global_step / (total_steps - 1)
    ratio = max(0., min(1., (4 / 7 - progress) / (2 / 7)))
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


class SpatialConv(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False, upsample=False):
        super().__init__()
        stride = 2 if downsample else 1
        conv = nn.Conv2d(in_channels, out_channels * (4 if upsample else 1),
                         3, stride=stride, padding=1)
        nn.init.trunc_normal_(conv.weight, std=0.02)
        nn.init.zeros_(conv.bias)
        self.conv = nn.Sequential(conv, nn.PixelShuffle(2)) if upsample else conv
        self.norm = nn.GroupNorm(2, out_channels)
        self.act = nn.SiLU()

    def forward(self, value):
        return self.act(self.norm(self.conv(value)))


class SpatialEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, depth):
        super().__init__()
        self.layers = nn.ModuleList([
            SpatialConv(in_channels if i == 0 else hidden_channels,
                        hidden_channels, downsample=bool(i % 2))
            for i in range(depth)]
        )

    def forward(self, value):
        skip = self.layers[0](value)
        value = skip
        for layer in self.layers[1:]:
            value = layer(value)
        return value, skip


class SpatialDecoder(nn.Module):
    def __init__(self, hidden_channels, out_channels, depth):
        super().__init__()
        self.layers = nn.ModuleList([
            SpatialConv(hidden_channels, hidden_channels, upsample=bool(i % 2))
            for i in reversed(range(depth))]
        )
        self.readout = nn.Conv2d(hidden_channels, out_channels, 1)

    def forward(self, value, skip, return_features=False):
        for layer in self.layers[:-1]:
            value = layer(value)
        features = value + skip
        if return_features:
            return features
        return self.readout(self.layers[-1](features))


class DropPath(nn.Module):
    def __init__(self, probability):
        super().__init__()
        self.probability = probability

    def forward(self, value):
        if not self.training or self.probability == 0:
            return value
        keep = 1 - self.probability
        shape = (value.shape[0],) + (1,) * (value.ndim - 1)
        mask = value.new_empty(shape).bernoulli_(keep).div_(keep)
        return value * mask


class TemporalAttention(nn.Module):
    """TAU large-kernel spatial gate with channel attention over stacked times."""

    def __init__(self, channels):
        super().__init__()
        self.proj_in = nn.Conv2d(channels, channels, 1)
        self.depthwise = nn.Conv2d(channels, channels, 5, padding=2, groups=channels)
        self.dilated = nn.Conv2d(channels, channels, 7, padding=9,
                                 dilation=3, groups=channels)
        self.spatial_projection = nn.Conv2d(channels, channels, 1)
        reduced = channels // max(channels // 16, 4)
        self.channel_gate = nn.Sequential(
            nn.Linear(channels, reduced, bias=False), nn.ReLU(),
            nn.Linear(reduced, channels, bias=False), nn.Sigmoid())
        self.proj_out = nn.Conv2d(channels, channels, 1)

    def forward(self, value):
        hidden = F.gelu(self.proj_in(value))
        spatial = self.spatial_projection(self.dilated(self.depthwise(hidden)))
        channel = self.channel_gate(hidden.mean((-2, -1)))[:, :, None, None]
        return value + self.proj_out(hidden * spatial * channel)


class TemporalAttentionUnit(nn.Module):
    def __init__(self, channels, mlp_ratio=8., dropout=0., drop_path=0.1):
        super().__init__()
        hidden = int(channels * mlp_ratio)
        self.norm1 = nn.BatchNorm2d(channels)
        self.norm2 = nn.BatchNorm2d(channels)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden),
            nn.GELU(), nn.Dropout(dropout), nn.Conv2d(hidden, channels, 1),
            nn.Dropout(dropout))
        # Preserve the original MLP initialization; TAU attention uses PyTorch defaults.
        for layer in self.mlp:
            if isinstance(layer, nn.Conv2d):
                fan_out = math.prod(layer.kernel_size) * layer.out_channels / layer.groups
                nn.init.normal_(layer.weight, std=math.sqrt(2 / fan_out))
                nn.init.zeros_(layer.bias)
        self.attention = TemporalAttention(channels)
        self.scale1 = nn.Parameter(torch.full((channels, 1, 1), 1e-2))
        self.scale2 = nn.Parameter(torch.full((channels, 1, 1), 1e-2))
        self.drop_path = DropPath(drop_path)

    def forward(self, value):
        value = value + self.drop_path(self.scale1 * self.attention(self.norm1(value)))
        return value + self.drop_path(self.scale2 * self.mlp(self.norm2(value)))


class TemporalTranslator(nn.Module):
    def __init__(self, seq_len, pred_len, channels, hidden_channels, depth,
                 mlp_ratio=8., dropout=0., drop_path=0.1):
        super().__init__()
        self.pred_len = pred_len
        widths = [seq_len * channels] + [hidden_channels] * (depth - 1)
        widths.append(pred_len * channels)
        rates = torch.linspace(0.01, drop_path, depth).tolist()
        self.blocks = nn.ModuleList([
            nn.Sequential(
                TemporalAttentionUnit(widths[i], mlp_ratio, dropout, rates[i]),
                nn.Conv2d(widths[i], widths[i + 1], 1)
                if widths[i] != widths[i + 1] else nn.Identity())
            for i in range(depth)])

    def forward(self, value):
        batch, steps, channels, height, width = value.shape
        value = value.reshape(batch, steps * channels, height, width)
        for block in self.blocks:
            value = block(value)
        return value.reshape(batch, self.pred_len, channels, height, width)


class SpatiotemporalEncoder(nn.Module):
    """Convolutional encoding and TAU translation to the forecast horizon."""

    def __init__(self, seq_len, pred_len, in_channels, channels, temporal_channels,
                 spatial_depth, temporal_depth, mlp_ratio, dropout, drop_path):
        super().__init__()
        self.pred_len = pred_len
        self.spatial = SpatialEncoder(in_channels, channels, spatial_depth)
        self.temporal = TemporalTranslator(
            seq_len, pred_len, channels, temporal_channels, temporal_depth,
            mlp_ratio, dropout, drop_path)
        self.skip_transfer = (nn.Conv2d(seq_len * channels, pred_len * channels, 1)
                              if seq_len != pred_len else nn.Identity())

    def forward(self, history):
        batch, steps = history.shape[:2]
        latent, skip = self.spatial(history.flatten(0, 1))
        channels, height, width = latent.shape[1:]
        future = self.temporal(latent.reshape(batch, steps, channels, height, width))
        sh, sw = skip.shape[-2:]
        skip = self.skip_transfer(skip.reshape(batch, steps * channels, sh, sw))
        return future, skip.reshape(batch * self.pred_len, channels, sh, sw)


class RegionalCloudPredictor(nn.Module):
    def __init__(self, seq_len, pred_len, in_channels, channels, temporal_channels,
                 spatial_depth, temporal_depth, mlp_ratio, dropout, drop_path):
        super().__init__()
        self.encoder = SpatiotemporalEncoder(
            seq_len, pred_len, in_channels, channels, temporal_channels,
            spatial_depth, temporal_depth, mlp_ratio, dropout, drop_path)
        self.decoder = SpatialDecoder(channels, in_channels, spatial_depth)

    def forward(self, history):
        future, skip = self.encoder(history)
        decoded = self.decoder(future.flatten(0, 1), skip)
        return decoded.reshape(*future.shape[:2], *decoded.shape[1:])


class LearnablePosition2D(nn.Module):
    def __init__(self, channels, height, width):
        super().__init__()
        self.row = nn.Parameter(torch.empty(1, channels // 2, height, 1))
        self.column = nn.Parameter(torch.empty(1, channels // 2, 1, width))
        nn.init.trunc_normal_(self.row, std=0.02)
        nn.init.trunc_normal_(self.column, std=0.02)

    def forward(self, value):
        height, width = value.shape[-2:]
        position = torch.cat((self.row.expand(-1, -1, -1, width),
                              self.column.expand(-1, -1, height, -1)), dim=1)
        return value + position


class CrossScaleSpatialFusion(nn.Module):
    """Local patch queries retrieve regional pixel tokens at each lead time."""

    def __init__(self, channels, height, width, roi_size, patch_size=2,
                 num_heads=4, dropout=0.):
        super().__init__()
        self.patch_size = patch_size
        self.input_projection = nn.Linear(2 * channels * patch_size**2, channels)
        self.output_projection = nn.Linear(channels, channels * patch_size**2)
        self.local_position = LearnablePosition2D(
            channels, height // patch_size, width // patch_size)
        self.regional_position = LearnablePosition2D(channels, *roi_size)
        self.local_norm = nn.LayerNorm(channels)
        self.regional_norm = nn.LayerNorm(channels)
        self.attention = nn.MultiheadAttention(
            channels, num_heads, dropout=dropout, batch_first=True
        )
        self.attention_scale = nn.Parameter(torch.full((channels,), 1e-2))
        self.ffn_norm = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(
            nn.Linear(channels, 4 * channels), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(4 * channels, channels), nn.Dropout(dropout))
        self.ffn_scale = nn.Parameter(torch.full((channels,), 1e-2))

    def forward(self, local, regional):
        batch, steps, channels, height, width = local.shape
        p = self.patch_size
        ph, pw = height // p, width // p
        patches = local.reshape(batch * steps, channels, ph, p, pw, p)
        patches = patches.permute(0, 2, 4, 1, 3, 5).reshape(
            batch * steps, ph, pw, channels * p * p)
        grid = self.input_projection(patches).permute(0, 3, 1, 2)
        query = self.local_position(grid).flatten(2).transpose(1, 2)
        context = self.regional_position(regional.flatten(0, 1)).flatten(2).transpose(1, 2)
        context = self.regional_norm(context)
        residual = self.attention(self.local_norm(query), context, context,
                                  need_weights=False)[0]
        fused = query + residual * self.attention_scale
        fused = fused + self.ffn(self.ffn_norm(fused)) * self.ffn_scale
        pixels = self.output_projection(fused).reshape(batch, steps, ph, pw, -1, p, p)
        return pixels.permute(0, 1, 4, 2, 5, 3, 6).reshape(
            batch, steps, -1, height, width)


class SpatiotemporalRefinementDecoder(nn.Module):
    def __init__(self, pred_len, channels, temporal_channels, spatial_depth,
                 temporal_depth, mlp_ratio, dropout):
        super().__init__()
        self.temporal = TemporalTranslator(
            pred_len, pred_len, channels, temporal_channels, temporal_depth,
            mlp_ratio, dropout, drop_path=0.)
        self.spatial = SpatialDecoder(channels, 3, spatial_depth)

    def forward(self, fused, skip, return_features=False):
        future = self.temporal(fused)
        output = self.spatial(future.flatten(0, 1), skip, return_features)
        return output.reshape(*future.shape[:2], *output.shape[1:])


def circumsolar_mse(prediction, target, mask, sigma_ratio=0.16):
    """Per-image Gaussian-weighted MSE; an empty solar mask contributes zero."""
    height, width = prediction.shape[-2:]
    mh, mw = mask.shape[-2:]
    weights = mask.detach().to(prediction)
    mass = weights.sum((-3, -2, -1))
    y = torch.arange(mh, device=weights.device, dtype=weights.dtype)
    x = torch.arange(mw, device=weights.device, dtype=weights.dtype)
    cy = (weights * y[None, None, None, :, None]).sum((-3, -2, -1)) / mass.clamp_min(1e-6)
    cx = (weights * x[None, None, None, None, :]).sum((-3, -2, -1)) / mass.clamp_min(1e-6)
    cy, cx = (cy + 0.5) * height / mh - 0.5, (cx + 0.5) * width / mw - 0.5
    yy = torch.arange(height, device=weights.device, dtype=weights.dtype)
    xx = torch.arange(width, device=weights.device, dtype=weights.dtype)
    distance = ((yy[None, None, :, None] - cy[..., None, None]).square()
                + (xx[None, None, None, :] - cx[..., None, None]).square())
    field = torch.exp(-distance / (2 * (sigma_ratio * min(height, width))**2))
    field = (field * (mass > 0)[..., None, None]).unsqueeze(2)
    numerator = ((prediction - target).square() * field).sum((-3, -2, -1))
    denominator = prediction.shape[2] * field.sum((-3, -2, -1))
    return (numerator / denominator.clamp_min(1e-6)).mean()


class CloudPredictor(nn.Module):
    """Stage 1, paper Eqs. regional--sky_decoder and objectives.

    forward(..., return_features=True) exposes Z_dec before the final spatial
    block and RGB readout. It is only used without image supervision in stage 2.
    """

    def __init__(self, seq_len_ground=3, seq_len_sate=6, pred_len=6,
                 ground_size=(128, 128), satellite_size=(128, 128),
                 channels_sate=2, local_channels=64, regional_channels=128,
                 temporal_channels=256, spatial_depth=4, regional_depth=8,
                 local_depth=8, refinement_depth=2, roi_encoder_depth=2,
                 roi_size=(8, 8), patch_size=2, num_heads=4, mlp_ratio=8.,
                 dropout=0., drop_path=0.1, lambda_sun=0.05, lambda_sate=1.,
                 sun_sigma_ratio=0.16):
        super().__init__()
        
        stride = 2 ** (spatial_depth // 2)
        self.pred_len, self.feature_channels = pred_len, local_channels
        self.ground_size, self.roi_size = tuple(ground_size), tuple(roi_size)
        self.ground_shape = (seq_len_ground, 3, *ground_size)
        self.satellite_shape = (seq_len_sate, channels_sate, *satellite_size)
        self.lambda_sun, self.lambda_sate = lambda_sun, lambda_sate
        self.sun_sigma_ratio = sun_sigma_ratio
        self.regional_predictor = RegionalCloudPredictor(
            seq_len_sate, pred_len, channels_sate, regional_channels,
            temporal_channels, spatial_depth, regional_depth, mlp_ratio,
            dropout, drop_path)
        self.local_encoder = SpatiotemporalEncoder(
            seq_len_ground, pred_len, 3, local_channels, temporal_channels,
            spatial_depth, local_depth, mlp_ratio, dropout, drop_path)
        self.solar_position_encoder = SpatialEncoder(1, local_channels, spatial_depth)
        self.roi_encoder = nn.Sequential(*[
            SpatialConv(channels_sate if i == 0 else local_channels, local_channels)
            for i in range(roi_encoder_depth)])
        self.spatial_fusion = CrossScaleSpatialFusion(
            local_channels, ground_size[0] // stride, ground_size[1] // stride,
            roi_size, patch_size, num_heads, dropout)
        self.refinement_decoder = SpatiotemporalRefinementDecoder(
            pred_len, local_channels, temporal_channels, spatial_depth,
            refinement_depth, mlp_ratio, dropout)

    def forward(self, ground_img_his, satellite_img_his, sun_mask_future,
                ground_img_future=None, satellite_img_future=None,
                teacher_forcing_ratio=0., return_features=False):
        batch = ground_img_his.shape[0]
        satellite = self.regional_predictor(satellite_img_his)
        condition = satellite.detach()
        ratio = teacher_forcing_ratio if self.training and satellite_img_future is not None else 0.
        if ratio == 1:
            condition = satellite_img_future
        elif ratio > 0:
            selected = torch.rand(batch, self.pred_len, 1, 1, 1, device=condition.device) < ratio
            condition = torch.where(selected, satellite_img_future, condition)

        local, skip = self.local_encoder(ground_img_his)
        mask = torch.nan_to_num(sun_mask_future.detach().to(local), nan=0.,
                                posinf=0., neginf=0.).clamp(0, 1)
        resized = F.interpolate(mask.flatten(0, 1), size=self.ground_size, mode="area")
        solar, _ = self.solar_position_encoder(resized)
        local = torch.cat((local, solar.reshape_as(local)), dim=2)
        rh, rw = self.roi_size
        y, x = (condition.shape[-2] - rh) // 2, (condition.shape[-1] - rw) // 2
        roi = condition[..., y:y + rh, x:x + rw]
        regional = self.roi_encoder(roi.flatten(0, 1))
        regional = regional.reshape(batch, self.pred_len, self.feature_channels, rh, rw)
        fused = self.spatial_fusion(local, regional)
        sky = self.refinement_decoder(fused, skip, return_features)

        loss, meta = None, {"teacher_forcing_ratio": ratio}
        if ground_img_future is not None:
            sky_loss = F.mse_loss(sky, ground_img_future)
            sun_loss = circumsolar_mse(sky, ground_img_future, mask, self.sun_sigma_ratio)
            loss = sky_loss + self.lambda_sun * sun_loss
            meta.update(sky_mse=sky_loss.detach(), circumsolar_mse=sun_loss.detach())
        if satellite_img_future is not None:
            satellite_loss = F.mse_loss(satellite, satellite_img_future)
            weighted = self.lambda_sate * satellite_loss
            loss = weighted if loss is None else loss + weighted
            meta["satellite_mse"] = satellite_loss.detach()
        return sky, satellite, loss, meta
